from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from pythia.capture import CapturePipeline


@dataclass(frozen=True)
class GroundingDINOConfig:
    detector_every_n_frames: int = 10
    buffer_size: int = 8
    target_capture_fps: float = 30.0
    preprocess_small_width: Optional[int] = None
    run_seconds: int = 10

    prompt: str = "button . text field . icon . container ."
    box_threshold: float = 0.25
    text_threshold: float = 0.25
    device: str = "cpu"

    model_config_path: str = ""
    model_checkpoint_path: str = ""

    ocr_lang: str = "en"
    ocr_conf_threshold: float = 0.3


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class _MacVisionOCREngine:
    def __init__(self, lang: str) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("macOS Vision OCR is only available on Darwin")
        if lang.lower() not in {"en", "eng"}:
            raise ValueError("Only English OCR is supported in this baseline")

        try:
            import Quartz
            import Vision
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "pyobjc Vision bindings are missing. Install: pip install pyobjc-framework-Vision"
            ) from e

        self._Quartz = Quartz
        self._Vision = Vision

    def _to_cgimage(self, rgb: np.ndarray):
        Quartz = self._Quartz

        h, w = rgb.shape[:2]
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[:, :, :3] = rgb
        rgba[:, :, 3] = 255

        data = rgba.tobytes()
        provider = Quartz.CGDataProviderCreateWithData(None, data, len(data), None)
        colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
        bits_per_component = 8
        bits_per_pixel = 32
        bytes_per_row = w * 4
        bitmap_info = Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaLast

        cgimage = Quartz.CGImageCreate(
            w,
            h,
            bits_per_component,
            bits_per_pixel,
            bytes_per_row,
            colorspace,
            bitmap_info,
            provider,
            None,
            False,
            Quartz.kCGRenderingIntentDefault,
        )
        return cgimage

    def run(self, rgb: np.ndarray) -> List[Dict[str, Any]]:
        Vision = self._Vision

        h, w = rgb.shape[:2]
        cgimage = self._to_cgimage(rgb)

        results_holder: Dict[str, Any] = {"results": None, "error": None}

        def _handler(request, error) -> None:  # noqa: ANN001
            results_holder["error"] = error
            results_holder["results"] = request.results()

        req = Vision.VNRecognizeTextRequest.alloc().initWithCompletionHandler_(_handler)
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(True)

        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cgimage, None)
        ok = handler.performRequests_error_([req], None)
        if not ok or results_holder["error"] is not None:
            return []

        out: List[Dict[str, Any]] = []
        results = results_holder["results"] or []

        for obs in results:
            candidates = obs.topCandidates_(1)
            if not candidates or len(candidates) == 0:
                continue
            cand = candidates[0]
            text = str(cand.string())
            conf = float(cand.confidence())

            bb = obs.boundingBox()
            x = float(bb.origin.x)
            y = float(bb.origin.y)
            bw = float(bb.size.width)
            bh = float(bb.size.height)

            x1 = int(x * w)
            x2 = int((x + bw) * w)
            y1 = int((1.0 - (y + bh)) * h)
            y2 = int((1.0 - y) * h)

            out.append(
                {
                    "class": "text",
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "confidence": conf,
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


def _import_groundingdino():
    try:
        import torch
        from PIL import Image

        from groundingdino.datasets import transforms as T
        from groundingdino.util.inference import load_model, predict

        return torch, Image, T, load_model, predict
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "GroundingDINO is not available in this Python environment. "
            "Run this script from a Python 3.11 venv where torch + GroundingDINO are installed. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e


def _prepare_image_tensor(rgb: np.ndarray, T, Image):  # noqa: ANN001
    pil = Image.fromarray(rgb)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    tensor, _ = transform(pil, None)
    return tensor


def _phrase_to_class(phrase: str) -> str:
    p = phrase.lower().strip()
    if "button" in p:
        return "button"
    if "field" in p or "input" in p or "textbox" in p:
        return "field"
    if "icon" in p:
        return "icon"
    if "container" in p or "panel" in p or "window" in p:
        return "container"
    if "text" in p or "label" in p:
        return "text"
    return "other"


def _run_dino_detector(rgb: np.ndarray, model, cfg: GroundingDINOConfig):
    torch, Image, T, _load_model, predict = _import_groundingdino()

    image_tensor = _prepare_image_tensor(rgb, T, Image)

    boxes, scores, phrases = predict(
        model=model,
        image=image_tensor,
        caption=cfg.prompt,
        box_threshold=float(cfg.box_threshold),
        text_threshold=float(cfg.text_threshold),
        device=str(cfg.device),
    )

    h, w = rgb.shape[:2]
    out: List[Dict[str, Any]] = []

    if boxes is None or len(boxes) == 0:
        return out

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    for (cx, cy, bw, bh), score, phrase in zip(boxes, scores, phrases):
        cxp = float(cx) * w
        cyp = float(cy) * h
        bwp = float(bw) * w
        bhp = float(bh) * h
        x1 = int(max(0.0, cxp - bwp / 2.0))
        y1 = int(max(0.0, cyp - bhp / 2.0))
        x2 = int(min(float(w - 1), cxp + bwp / 2.0))
        y2 = int(min(float(h - 1), cyp + bhp / 2.0))

        out.append(
            {
                "class": _phrase_to_class(str(phrase)),
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
                "source": "groundingdino",
                "phrase": str(phrase),
            }
        )

    return out


def run_groundingdino_step2(
    *,
    pipeline: CapturePipeline,
    cfg: GroundingDINOConfig,
    output_dir: Path,
    run_name: str,
) -> Dict[str, Any]:
    torch, Image, T, load_model, _predict = _import_groundingdino()

    if not cfg.model_config_path or not cfg.model_checkpoint_path:
        raise ValueError("model_config_path and model_checkpoint_path are required")

    _safe_mkdir(output_dir)

    model = load_model(cfg.model_config_path, cfg.model_checkpoint_path, device=str(cfg.device))

    ocr_engine = None
    try:
        ocr_engine = _MacVisionOCREngine(lang=cfg.ocr_lang)
    except Exception:
        ocr_engine = None

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
        ui = _run_dino_detector(frame.rgb, model, cfg)
        t1 = time.perf_counter()

        ocr: List[Dict[str, Any]] = []
        if ocr_engine is not None:
            ocr_all = ocr_engine.run(frame.rgb)
            ocr = [x for x in ocr_all if x.get("confidence", 0.0) >= cfg.ocr_conf_threshold]
        t2 = time.perf_counter()

        overlay_ui = _draw_overlay(frame.rgb, ui, [])
        overlay_ocr = _draw_overlay(frame.rgb, [], ocr)
        overlay_both = _draw_overlay(frame.rgb, ui, ocr)

        latency_ms = max(0.0, (time.perf_counter() - frame.ts_monotonic) * 1000.0)
        latencies_ms.append(latency_ms)

        stamp = _now_ms()
        json_path = output_dir / f"ui_state_{stamp}.json"
        img_ui_path = output_dir / f"overlay_ui_{stamp}.jpg"
        img_ocr_path = output_dir / f"overlay_ocr_{stamp}.jpg"
        img_both_path = output_dir / f"overlay_{stamp}.jpg"

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
                "prompt": cfg.prompt,
                "box_threshold": cfg.box_threshold,
                "text_threshold": cfg.text_threshold,
                "device": cfg.device,
            },
            "timing_ms": {
                "ui_detector": (t1 - t0) * 1000.0,
                "ocr": (t2 - t1) * 1000.0,
                "e2e_latency": latency_ms,
            },
            "elements": ui + ocr,
        }

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        cv2.imwrite(str(img_ui_path), cv2.cvtColor(overlay_ui, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img_ocr_path), cv2.cvtColor(overlay_ocr, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img_both_path), cv2.cvtColor(overlay_both, cv2.COLOR_RGB2BGR))

        artifacts.append(
            {
                "ui_state": str(json_path),
                "overlay_ui": str(img_ui_path),
                "overlay_ocr": str(img_ocr_path),
                "overlay": str(img_both_path),
            }
        )

    elapsed = time.perf_counter() - start
    produced_total = pipeline.buffer.stats().produced - capture_stats0

    capture_fps = (produced_total / elapsed) if elapsed > 0 else 0.0
    detector_fps = (processed / elapsed) if elapsed > 0 else 0.0
    ocr_fps = detector_fps
    latency_avg = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0

    return {
        "date": time.strftime("%Y-%m-%d"),
        "stage": "Step2.B (GroundingDINO + OCR)",
        "config": (
            f"resolution={os.getenv('PYTHIA_RES', 'orig')} process_size=orig "
            f"detector_every_n_frames={cfg.detector_every_n_frames} ocr_every_n_frames={cfg.detector_every_n_frames} "
            f"buffer={cfg.buffer_size} box_th={cfg.box_threshold} text_th={cfg.text_threshold} "
            f"device={cfg.device}"
        ),
        "capture_fps": capture_fps,
        "detector_fps": detector_fps,
        "ocr_fps": ocr_fps,
        "e2e_latency_ms": latency_avg,
        "artifacts": artifacts[:3],
    }
