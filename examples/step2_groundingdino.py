from __future__ import annotations

import argparse
import time
from pathlib import Path

from pythia.capture import CapturePipeline, FrameBuffer, PreprocessConfig, ScreenSource
from pythia.vision import GroundingDINOConfig, run_groundingdino_step2


def _append_benchmark_row(path: Path, row: dict) -> None:
    line = (
        f"| {row['date']} | {row['stage']} | {row['config']} | "
        f"{row['capture_fps']:.2f} | {row['detector_fps']:.2f} | {row['ocr_fps']:.2f} | {row['e2e_latency_ms']:.1f} | "
        f"{row['artifacts'][0]['overlay'] if row['artifacts'] else ''} |\n"
    )
    path.write_text(path.read_text() + line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--every-n", type=int, default=10)
    parser.add_argument("--buffer", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--small-width", type=int, default=0)

    parser.add_argument("--model-config", required=True)
    parser.add_argument("--model-ckpt", required=True)
    parser.add_argument("--device", default="cpu")

    parser.add_argument("--prompt", default="button . text field . icon . container .")
    parser.add_argument("--box-th", type=float, default=0.25)
    parser.add_argument("--text-th", type=float, default=0.25)

    parser.add_argument("--ocr-conf", type=float, default=0.3)
    args = parser.parse_args()

    run_name = args.run_name or time.strftime("step2B_%Y%m%d_%H%M%S")
    out_dir = Path("benchmarks") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    source = ScreenSource(target_fps=args.fps)
    buffer = FrameBuffer(maxlen=args.buffer)
    preprocess = PreprocessConfig(small_width=args.small_width if args.small_width > 0 else None)
    pipeline = CapturePipeline(source=source, buffer=buffer, preprocess=preprocess)
    pipeline.start()

    cfg = GroundingDINOConfig(
        detector_every_n_frames=max(1, args.every_n),
        buffer_size=args.buffer,
        target_capture_fps=args.fps,
        preprocess_small_width=preprocess.small_width,
        run_seconds=args.seconds,
        prompt=args.prompt,
        box_threshold=args.box_th,
        text_threshold=args.text_th,
        device=args.device,
        model_config_path=args.model_config,
        model_checkpoint_path=args.model_ckpt,
        ocr_lang="en",
        ocr_conf_threshold=args.ocr_conf,
    )

    try:
        row = run_groundingdino_step2(pipeline=pipeline, cfg=cfg, output_dir=out_dir, run_name=run_name)
    finally:
        pipeline.stop()

    bench_path = Path("Benchmarks_along_the_way.md")
    _append_benchmark_row(bench_path, row)

    print(f"Saved artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
