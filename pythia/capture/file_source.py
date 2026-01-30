from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2

from .types import RawFrame


class FileSource:
    def __init__(self, path: str, realtime: bool = True, target_fps: Optional[float] = None) -> None:
        self._path = path
        self._realtime = realtime
        self._target_fps = target_fps
        self._cap: Optional[cv2.VideoCapture] = None
        self._closed = False

    def frames(self) -> Iterator[RawFrame]:
        if self._closed:
            return

        self._cap = cv2.VideoCapture(self._path)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video file: {self._path}")

        fps = self._cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        if self._target_fps and self._target_fps > 0:
            fps = self._target_fps

        period = 1.0 / fps
        next_deadline = time.perf_counter()

        while not self._closed:
            ok, bgr = self._cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            ts_m = time.perf_counter()
            ts_w = time.time()
            h, w = rgb.shape[:2]

            yield RawFrame(ts_monotonic=ts_m, ts_wall=ts_w, width=w, height=h, rgb=rgb)

            if self._realtime:
                now = time.perf_counter()
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                next_deadline = max(next_deadline + period, time.perf_counter())

    def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None
