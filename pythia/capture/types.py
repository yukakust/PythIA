from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class RawFrame:
    ts_monotonic: float
    ts_wall: float
    width: int
    height: int
    rgb: np.ndarray


@dataclass(frozen=True)
class Frame:
    ts_monotonic: float
    ts_wall: float
    width: int
    height: int
    rgb: np.ndarray
    small_rgb: Optional[np.ndarray]
    small_size: Optional[Tuple[int, int]]
    scale_x: float
    scale_y: float
