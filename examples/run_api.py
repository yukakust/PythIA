from __future__ import annotations

import argparse
import uvicorn

from pythia.api import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--buffer", type=int, default=8)
    parser.add_argument("--small-width", type=int, default=960)
    args = parser.parse_args()

    app = create_app(
        target_fps=args.fps,
        buffer_size=args.buffer,
        small_width=args.small_width if args.small_width > 0 else None,
    )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
