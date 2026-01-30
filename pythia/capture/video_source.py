from __future__ import annotations

from typing import Iterator, Protocol

from .types import RawFrame


class VideoSource(Protocol):
    def frames(self) -> Iterator[RawFrame]:
        ...

    def close(self) -> None:
        ...
