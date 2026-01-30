from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterator, Optional

from .types import Frame


@dataclass(frozen=True)
class FrameStats:
    produced: int
    dropped: int


class FrameBuffer:
    def __init__(self, maxlen: int = 8) -> None:
        if maxlen <= 0:
            raise ValueError("maxlen must be > 0")

        self._buf: Deque[Frame] = deque(maxlen=maxlen)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._closed = False
        self._version = 0

        self._produced = 0
        self._dropped = 0

    def put(self, frame: Frame) -> None:
        with self._cv:
            if self._closed:
                return

            if len(self._buf) == self._buf.maxlen:
                self._dropped += 1

            self._buf.append(frame)
            self._produced += 1
            self._version += 1
            self._cv.notify_all()

    def get_latest(self) -> Optional[Frame]:
        with self._lock:
            if not self._buf:
                return None
            return self._buf[-1]

    def stats(self) -> FrameStats:
        with self._lock:
            return FrameStats(produced=self._produced, dropped=self._dropped)

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def frames(self) -> Iterator[Frame]:
        last_version = -1
        while True:
            with self._cv:
                while not self._closed and self._version == last_version:
                    self._cv.wait()

                if self._closed:
                    return

                if not self._buf:
                    continue

                frame = self._buf[-1]
                last_version = self._version

            yield frame
