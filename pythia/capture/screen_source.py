from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np
import mss

from .types import RawFrame


@dataclass(frozen=True)
class ScreenRegion:
    left: int
    top: int
    width: int
    height: int


class ScreenSource:
    def __init__(
        self,
        monitor_index: int = 1,
        region: Optional[ScreenRegion] = None,
        target_fps: Optional[float] = None,
    ) -> None:
        self._monitor_index = monitor_index
        self._region = region
        self._target_fps = target_fps
        self._sct: Optional[mss.mss] = None
        self._closed = False

    def frames(self) -> Iterator[RawFrame]:
        if self._closed:
            return

        self._sct = mss.mss()

        if self._region is None:
            mon = self._sct.monitors[self._monitor_index]
            grab_region = {
                "left": mon["left"],
                "top": mon["top"],
                "width": mon["width"],
                "height": mon["height"],
            }
        else:
            grab_region = {
                "left": self._region.left,
                "top": self._region.top,
                "width": self._region.width,
                "height": self._region.height,
            }

        period = None
        if self._target_fps and self._target_fps > 0:
            period = 1.0 / self._target_fps

        next_deadline = time.perf_counter()
        while not self._closed:
            if period is not None:
                now = time.perf_counter()
                if now < next_deadline:
                    time.sleep(next_deadline - now)
                next_deadline = max(next_deadline + period, time.perf_counter())

            img = self._sct.grab(grab_region)

            frame_rgba = np.asarray(img, dtype=np.uint8)
            bgr = frame_rgba[:, :, :3]
            rgb = bgr[:, :, ::-1].copy()

            ts_m = time.perf_counter()
            ts_w = time.time()
            h, w = rgb.shape[:2]

            yield RawFrame(
                ts_monotonic=ts_m,
                ts_wall=ts_w,
                width=w,
                height=h,
                rgb=rgb,
            )

    def close(self) -> None:
        self._closed = True
        if self._sct is not None:
            self._sct.close()
            self._sct = None
