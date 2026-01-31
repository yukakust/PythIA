from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from PIL import Image
import torch

from pythia.capture import CapturePipeline
from pythia.vision.baseline import _MacVisionOCREngine, _draw_overlay  # noqa: PLC2701


@dataclass(frozen=True)
class OmniParserConfig:
    detector_every_n_frames: int = 10
    buffer_size: int = 8
    target_capture_fps: float = 30.0
    preprocess_small_width: Optional[int] = None
    run_seconds: int = 10
    max_frames: Optional[int] = None

    omni_path: str = ""
    mode: int = 2  # 2: UI-only + our OCR, 3: UI + Omni content + our OCR, 1: Omni UI + Omni OCR/content

    box_threshold: float = 0.05
    iou_threshold: float = 0.7
    imgsz: int = 640

    caption_batch_size: int = 8

    ocr_lang: str = "en"
    ocr_conf_threshold: float = 0.3

    omni_use_paddleocr: bool = False


def _now_ms() -> int:
    return int(time.time() * 1000)


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _load_omniparser_utils_without_paddleocr(omni_path: str):
    utils_path = Path(omni_path) / "util" / "utils.py"
    if not utils_path.exists():
        raise RuntimeError(f"OmniParser utils.py not found: {utils_path}")

    src = utils_path.read_text(encoding="utf-8")

    # OmniParser initializes PaddleOCR at import time, which is brittle across PaddleOCR versions.
    # For our modes (2/3) we do not use PaddleOCR at all, so we strip that initialization.
    src = src.replace("from paddleocr import PaddleOCR", "PaddleOCR = None")

    # Remove the whole block:
    # paddle_ocr = PaddleOCR(...)
    src = re.sub(
        r"\n\s*paddle_ocr\s*=\s*PaddleOCR\(.*?\)\s*\n",
        "\n# paddle_ocr init stripped by PythIA\n",
        src,
        flags=re.DOTALL,
    )

    # Also remove easyocr reader init if desired later; for now it is harmless.

    import types

    mod = types.ModuleType("omniparser_utils_no_paddleocr")
    mod.__file__ = str(utils_path)
    # Ensure OmniParser package imports (util.*) resolve.
    if omni_path not in sys.path:
        sys.path.insert(0, omni_path)
    exec(compile(src, str(utils_path), "exec"), mod.__dict__)
    return mod


def _import_omniparser_utils(omni_path: str):
    if not omni_path:
        raise ValueError("omni_path is required (path to OmniParser repo folder)")

    # OmniParser uses relative imports like: from util.utils import ...
    if omni_path not in sys.path:
        sys.path.insert(0, omni_path)

    try:
        from util.utils import (  # type: ignore
            check_ocr_box,
            get_caption_model_processor,
            get_som_labeled_img,
            get_yolo_model,
        )

        return get_yolo_model, get_caption_model_processor, get_som_labeled_img, check_ocr_box
    except Exception as e:  # pragma: no cover
        # Known OmniParser issue: PaddleOCR init fails at import time depending on PaddleOCR version.
        # For our mode=2/3 we don't need PaddleOCR, so load a patched module.
        msg = str(e)
        if (
            "use_dilation" in msg
            or "show_log" in msg
            or "Unknown argument" in msg
            or "PaddleOCR" in msg
        ):
            patched = _load_omniparser_utils_without_paddleocr(omni_path)
            return (
                patched.get_yolo_model,
                patched.get_caption_model_processor,
                patched.get_som_labeled_img,
                patched.check_ocr_box,
            )

        raise RuntimeError(
            "Failed to import OmniParser. Make sure OmniParser repo exists at omni_path and you're running from an env "
            "where its requirements are installed. "
            f"Original error: {type(e).__name__}: {e}"
        ) from e


def _ocr_to_ratio_xyxy(ocr: List[Dict[str, Any]], w: int, h: int):
    boxes: List[List[float]] = []
    texts: List[str] = []
    for el in ocr:
        x1, y1, x2, y2 = el["bbox"]
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append([x1 / w, y1 / h, x2 / w, y2 / h])
        texts.append(str(el.get("text", "")))
    # Important: OmniParser code assumes ocr_bbox is truthy when used; use empty list instead of None.
    return boxes, texts


def _map_omni_type_to_class(t: str) -> str:
    tt = str(t).lower().strip()
    if tt in {"icon", "text"}:
        return tt
    return "other"


def _omni_boxes_to_ui_elements(
    *,
    boxes: List[Dict[str, Any]],
    w: int,
    h: int,
    include_content: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for el in boxes:
        src = str(el.get("source", ""))
        if src == "box_ocr_content_ocr":
            continue

        bbox = el.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        # OmniParser returns ratio coords when output_coord_in_ratio=True
        x1 = int(float(bbox[0]) * w)
        y1 = int(float(bbox[1]) * h)
        x2 = int(float(bbox[2]) * w)
        y2 = int(float(bbox[3]) * h)
        if x2 <= x1 or y2 <= y1:
            continue

        item: Dict[str, Any] = {
            "class": _map_omni_type_to_class(el.get("type")),
            "bbox": [x1, y1, x2, y2],
            "confidence": 0.5,
            "source": "omniparser",
        }

        if include_content:
            content = el.get("content")
            if content is not None:
                item["content"] = str(content)

        out.append(item)

    return out


def run_omniparser_step2(
    *,
    pipeline: CapturePipeline,
    cfg: OmniParserConfig,
    output_dir: Path,
    run_name: str,
) -> Dict[str, Any]:
    _safe_mkdir(output_dir)

    get_yolo_model, get_caption_model_processor, get_som_labeled_img, check_ocr_box = _import_omniparser_utils(cfg.omni_path)

    # UI detector
    yolo_model = get_yolo_model(model_path=str(Path(cfg.omni_path) / "weights" / "icon_detect" / "model.pt"))

    include_content = cfg.mode in {1, 3}
    use_local_semantics = cfg.mode in {1, 3}

    caption_model_processor = None
    if use_local_semantics:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        caption_model_processor = get_caption_model_processor(
            model_name="florence2",
            model_name_or_path=str(Path(cfg.omni_path) / "weights" / "icon_caption_florence"),
            device=device,
        )

    ocr_engine = None
    if cfg.mode in {2, 3}:
        ocr_engine = _MacVisionOCREngine(lang=cfg.ocr_lang)

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

        # OCR first (we pass boxes to OmniParser for overlap removal)
        ocr: List[Dict[str, Any]] = []
        ocr_ratio_boxes: List[List[float]] = []
        ocr_texts: List[str] = []
        if cfg.mode in {2, 3} and ocr_engine is not None:
            ocr_all = ocr_engine.run(frame.rgb)
            ocr = [x for x in ocr_all if x.get("confidence", 0.0) >= cfg.ocr_conf_threshold]
            ocr_ratio_boxes, ocr_texts = _ocr_to_ratio_xyxy(ocr, frame.width, frame.height)
        elif cfg.mode == 1:
            texts, bb = check_ocr_box(
                Image.fromarray(frame.rgb),
                display_img=False,
                output_bb_format="xyxy",
                easyocr_args={"text_threshold": float(cfg.ocr_conf_threshold)},
                use_paddleocr=True,
            )[0]
            for t, b in zip(texts, bb):
                x, y, x2, y2 = b
                if x2 <= x or y2 <= y:
                    continue
                ocr.append(
                    {
                        "class": "text",
                        "bbox": [int(x), int(y), int(x2), int(y2)],
                        "text": str(t),
                        "confidence": 0.5,
                        "source": "omni_ocr",
                    }
                )
            ocr_ratio_boxes = [[float(x1), float(y1), float(x2), float(y2)] for x1, y1, x2, y2 in [x["bbox"] for x in ocr]]
            ocr_texts = [str(x.get("text", "")) for x in ocr]

        # OmniParser core
        _, _, parsed_content_list = get_som_labeled_img(
            Image.fromarray(frame.rgb),
            yolo_model,
            BOX_TRESHOLD=float(cfg.box_threshold),
            output_coord_in_ratio=True,
            ocr_bbox=ocr_ratio_boxes,
            ocr_text=ocr_texts,
            use_local_semantics=use_local_semantics,
            iou_threshold=float(cfg.iou_threshold),
            scale_img=False,
            imgsz=int(cfg.imgsz),
            caption_model_processor=caption_model_processor,
            batch_size=int(cfg.caption_batch_size),
        )

        ui = _omni_boxes_to_ui_elements(
            boxes=parsed_content_list,
            w=frame.width,
            h=frame.height,
            include_content=include_content,
        )
        t1 = time.perf_counter()

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
                "mode": cfg.mode,
                "max_frames": cfg.max_frames,
                "box_threshold": cfg.box_threshold,
                "iou_threshold": cfg.iou_threshold,
                "imgsz": cfg.imgsz,
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
                "json": str(json_path),
                "overlay_ui": str(img_ui_path),
                "overlay_ocr": str(img_ocr_path),
                "overlay": str(img_both_path),
            }
        )

        if cfg.max_frames is not None and processed >= int(cfg.max_frames):
            break

    elapsed = time.perf_counter() - start
    produced_total = pipeline.buffer.stats().produced - capture_stats0

    capture_fps = (produced_total / elapsed) if elapsed > 0 else 0.0
    detector_fps = (processed / elapsed) if elapsed > 0 else 0.0
    ocr_fps = detector_fps
    latency_avg = float(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0

    stage = "Step2.C (OmniParser)"
    if cfg.mode == 2:
        stage = "Step2.C (OmniParser UI + macOS OCR)"
    elif cfg.mode == 3:
        stage = "Step2.C (OmniParser UI+content + macOS OCR)"

    return {
        "date": time.strftime("%Y-%m-%d"),
        "stage": stage,
        "config": (
            f"resolution={os.getenv('PYTHIA_RES', 'orig')} process_size=orig "
            f"detector_every_n_frames={cfg.detector_every_n_frames} ocr_every_n_frames={cfg.detector_every_n_frames} "
            f"buffer={cfg.buffer_size} mode={cfg.mode} box_th={cfg.box_threshold} iou_th={cfg.iou_threshold} imgsz={cfg.imgsz}"
        ),
        "capture_fps": capture_fps,
        "detector_fps": detector_fps,
        "ocr_fps": ocr_fps,
        "e2e_latency_ms": latency_avg,
        "artifacts": artifacts[:3],
    }
