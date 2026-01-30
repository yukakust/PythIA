from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2

from .types import Frame, RawFrame


@dataclass(frozen=True)
class PreprocessConfig:
    small_width: Optional[int] = 960


def preprocess_frame(raw: RawFrame, cfg: PreprocessConfig) -> Frame:
    small_rgb = None
    small_size: Optional[Tuple[int, int]] = None
    scale_x = 1.0
    scale_y = 1.0

    if cfg.small_width is not None and cfg.small_width > 0 and raw.width > cfg.small_width:
        new_w = int(cfg.small_width)
        new_h = int(round(raw.height * (new_w / raw.width)))
        small_rgb = cv2.resize(raw.rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        small_size = (new_w, new_h)
        scale_x = raw.width / new_w
        scale_y = raw.height / new_h

    return Frame(
        ts_monotonic=raw.ts_monotonic,
        ts_wall=raw.ts_wall,
        width=raw.width,
        height=raw.height,
        rgb=raw.rgb,
        small_rgb=small_rgb,
        small_size=small_size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
