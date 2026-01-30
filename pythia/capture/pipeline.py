from __future__ import annotations

import threading
from typing import Iterator, Optional

from .frame_buffer import FrameBuffer
from .preprocess import PreprocessConfig, preprocess_frame
from .types import Frame
from .video_source import VideoSource


class CapturePipeline:
    def __init__(
        self,
        source: VideoSource,
        buffer: Optional[FrameBuffer] = None,
        preprocess: Optional[PreprocessConfig] = None,
    ) -> None:
        self._source = source
        self._buffer = buffer or FrameBuffer()
        self._preprocess = preprocess or PreprocessConfig()

        self._thread: Optional[threading.Thread] = None
        self._stop_evt = threading.Event()

    @property
    def buffer(self) -> FrameBuffer:
        return self._buffer

    def start(self) -> None:
        if self._thread is not None:
            return

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="capture-pipeline", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            for raw in self._source.frames():
                if self._stop_evt.is_set():
                    break
                frame = preprocess_frame(raw, self._preprocess)
                self._buffer.put(frame)
        finally:
            self._buffer.close()
            self._source.close()

    def stop(self) -> None:
        self._stop_evt.set()
        self._source.close()
        self._buffer.close()

    def get_latest(self) -> Optional[Frame]:
        return self._buffer.get_latest()

    def frames(self) -> Iterator[Frame]:
        return self._buffer.frames()

    def frames_all(self, *, start_from_latest: bool = False) -> Iterator[Frame]:
        return self._buffer.frames_all(start_from_latest=start_from_latest)
