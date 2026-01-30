from __future__ import annotations

import argparse
import time

import cv2

from pythia.capture import CapturePipeline, PreprocessConfig, ScreenSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--small-width", type=int, default=960)
    parser.add_argument("--buffer", type=int, default=8)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    source = ScreenSource(target_fps=args.fps)
    pipeline = CapturePipeline(
        source=source,
        preprocess=PreprocessConfig(small_width=args.small_width if args.small_width > 0 else None),
    )
    pipeline.start()

    t0 = time.perf_counter()
    frames = 0

    try:
        for frame in pipeline.frames():
            frames += 1
            now = time.perf_counter()
            if now - t0 >= 1.0:
                stats = pipeline.buffer.stats()
                fps_est = frames / (now - t0)
                print(f"fps={fps_est:.1f} produced={stats.produced} dropped={stats.dropped}")
                t0 = now
                frames = 0

            if args.show:
                img = frame.small_rgb if frame.small_rgb is not None else frame.rgb
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imshow("PythIA capture", bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
