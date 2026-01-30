from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pythia.capture import CapturePipeline, Frame


@dataclass(frozen=True)
class BaselineConfig:
    detector_every_n_frames: int = 10
    buffer_size: int = 8
    target_capture_fps: float = 30.0
    preprocess_small_width: Optional[int] = None
    run_seconds: int = 10
    ocr_lang: str = "en"
    ocr_conf_threshold: float = 0.3


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _detect_ui_heuristics(rgb: np.ndarray) -> List[Dict[str, Any]]:
    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw <= 0 or bh <= 0:
            continue

        area = bw * bh
        if area < 400:
            continue

        if bw > int(0.98 * w) and bh > int(0.98 * h):
            continue

        ar = bw / float(bh)

        cls = "other"
        if area > (w * h) * 0.15:
            cls = "container"
        elif bw <= 80 and bh <= 80:
            cls = "icon"
        elif bh <= 60 and bw >= 140 and ar >= 2.2:
            cls = "field"
        elif 0.5 <= ar <= 6.0 and 25 <= bh <= 90:
            cls = "button"

        out.append(
            {
                "class": cls,
                "bbox": [int(x), int(y), int(x + bw), int(y + bh)],
                "confidence": 0.5,
                "source": "heuristic",
            }
        )

    out_sorted: List[Dict[str, Any]] = []
    for item in sorted(out, key=lambda it: (it["bbox"][1], it["bbox"][0])):
        bbox = tuple(item["bbox"])
        keep = True
        for kept in out_sorted:
            if _iou(bbox, tuple(kept["bbox"])) > 0.85:
                keep = False
                break
        if keep:
            out_sorted.append(item)

    return out_sorted


class _RapidOCREngine:
    def __init__(self, lang: str) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "rapidocr-onnxruntime is not installed. Install it with: pip3 install rapidocr-onnxruntime"
            ) from e

        self._engine = RapidOCR(lang=lang)

    def run(self, bgr: np.ndarray) -> List[Dict[str, Any]]:
        res, _ = self._engine(bgr)
        out: List[Dict[str, Any]] = []
        if not res:
            return out

        for item in res:
            poly, text, conf = item
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
            out.append(
                {
                    "class": "text",
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "confidence": float(conf),
                    "source": "ocr",
                }
            )

        return out


def _draw_overlay(rgb: np.ndarray, ui: List[Dict[str, Any]], ocr: List[Dict[str, Any]]) -> np.ndarray:
    img = rgb.copy()

    for el in ui:
        x1, y1, x2, y2 = el["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            img,
            el["class"],
            (x1, max(10, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    for el in ocr:
        x1, y1, x2, y2 = el["bbox"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        txt = el.get("text")
        if txt:
            cv2.putText(
                img,
                str(txt)[:32],
                (x1, min(img.shape[0] - 5, y2 + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return img


def run_baseline_step2(
    *,
    pipeline: CapturePipeline,
    cfg: BaselineConfig,
    output_dir: Path,
    run_name: str,
) -> Dict[str, Any]:
    _safe_mkdir(output_dir)

    ocr_engine = _RapidOCREngine(lang=cfg.ocr_lang)

    start = time.perf_counter()
    capture_stats0 = pipeline.buffer.stats().produced
    processed = 0
    latencies_ms: List[float] = []

    frame_idx = 0
    artifacts: List[Dict[str, str]] = []

    for frame in pipeline.frames_all(start_from_latest=True):
        now = time.perf_counter()
        if now - start >= cfg.run_seconds:
            break

        frame_idx += 1
        if cfg.detector_every_n_frames > 1 and (frame_idx % cfg.detector_every_n_frames) != 0:
            continue

        processed += 1

        t0 = time.perf_counter()
        ui = _detect_ui_heuristics(frame.rgb)
        t1 = time.perf_counter()

        bgr = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        ocr_all = ocr_engine.run(bgr)
        ocr = [x for x in ocr_all if x.get("confidence", 0.0) >= cfg.ocr_conf_threshold]
        t2 = time.perf_counter()

        overlay = _draw_overlay(frame.rgb, ui, ocr)

        latency_ms = max(0.0, (time.perf_counter() - frame.ts_monotonic) * 1000.0)
        latencies_ms.append(latency_ms)

        stamp = _now_ms()
        json_path = output_dir / f"ui_state_{stamp}.json"
        img_path = output_dir / f"overlay_{stamp}.jpg"

        payload = {
            "run": {
                "name": run_name,
                "ts_wall": frame.ts_wall,
                "ts_monotonic": frame.ts_monotonic,
            },
            "frame": {
                "width": frame.width,
                "height": frame.height,
                "process_width": frame.width,
                "process_height": frame.height,
            },
            "config": {
                "detector_every_n_frames": cfg.detector_every_n_frames,
                "ocr_every_n_frames": cfg.detector_every_n_frames,
                "buffer": cfg.buffer_size,
            },
            "timing_ms": {
                "ui_detector": (t1 - t0) * 1000.0,
                "ocr": (t2 - t1) * 1000.0,
                "e2e_latency": latency_ms,
            },
            "elements": ui + ocr,
        }

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        cv2.imwrite(str(img_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        artifacts.append({"ui_state": str(json_path), "overlay": str(img_path)})

    elapsed = time.perf_counter() - start
    produced_total = pipeline.buffer.stats().produced - capture_stats0

    capture_fps = (produced_total / elapsed) if elapsed > 0 else 0.0
    detector_fps = (processed / elapsed) if elapsed > 0 else 0.0
    ocr_fps = detector_fps
    latency_avg = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0

    return {
        "date": time.strftime("%Y-%m-%d"),
        "stage": "Step2.A (baseline heuristics + OCR)",
        "config": f"resolution={os.getenv('PYTHIA_RES', 'orig')} process_size=orig detector_every_n_frames={cfg.detector_every_n_frames} ocr_every_n_frames={cfg.detector_every_n_frames} buffer={cfg.buffer_size}",
        "capture_fps": capture_fps,
        "detector_fps": detector_fps,
        "ocr_fps": ocr_fps,
        "e2e_latency_ms": latency_avg,
        "artifacts": artifacts[:3],
    }
