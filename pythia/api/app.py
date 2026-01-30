from __future__ import annotations

import io
import threading
import time
from dataclasses import asdict
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query, Response
from PIL import Image

from pythia.capture import CapturePipeline, Frame, FrameBuffer, PreprocessConfig, ScreenSource


class _FrameRate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0 = time.perf_counter()
        self._last_produced = 0
        self._last_fps: Optional[float] = None

    def update(self, produced: int) -> None:
        with self._lock:
            now = time.perf_counter()
            dt = now - self._t0
            if dt >= 1.0:
                dp = produced - self._last_produced
                self._last_fps = dp / dt if dt > 0 else None
                self._t0 = now
                self._last_produced = produced

    def get(self) -> Optional[float]:
        with self._lock:
            return self._last_fps


def _encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def create_app(
    *,
    target_fps: float = 30.0,
    buffer_size: int = 8,
    small_width: Optional[int] = 960,
) -> FastAPI:
    app = FastAPI(title="PythIA API")

    source = ScreenSource(target_fps=target_fps)
    buffer = FrameBuffer(maxlen=buffer_size)
    pipeline = CapturePipeline(source=source, buffer=buffer, preprocess=PreprocessConfig(small_width=small_width))
    fps_meter = _FrameRate()

    def _get_latest() -> Optional[Frame]:
        return pipeline.get_latest()

    @app.on_event("startup")
    def _startup() -> None:
        pipeline.start()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        pipeline.stop()

    @app.get("/latest_frame_meta")
    def latest_frame_meta() -> dict:
        frame = _get_latest()
        stats = pipeline.buffer.stats()
        fps_meter.update(stats.produced)
        return {
            "ok": frame is not None,
            "frame": None
            if frame is None
            else {
                "ts_monotonic": frame.ts_monotonic,
                "ts_wall": frame.ts_wall,
                "width": frame.width,
                "height": frame.height,
                "small_size": frame.small_size,
                "scale_x": frame.scale_x,
                "scale_y": frame.scale_y,
            },
            "buffer": asdict(stats),
            "fps": fps_meter.get(),
        }

    @app.get("/latest_frame_jpeg")
    def latest_frame_jpeg(
        small: bool = Query(default=True),
        quality: int = Query(default=80, ge=10, le=95),
    ) -> Response:
        frame = _get_latest()
        if frame is None:
            return Response(status_code=404)

        rgb = frame.small_rgb if (small and frame.small_rgb is not None) else frame.rgb
        data = _encode_jpeg(rgb, quality=quality)
        return Response(content=data, media_type="image/jpeg")

    return app
