# Benchmarks_along_the_way

This document is a running log of performance measurements taken during development.

## Metric definitions
- **capture_fps**: frames/sec produced by the capture pipeline
- **detector_fps**: frames/sec processed by the UI detector stage
- **ocr_fps**: frames/sec processed by the OCR stage
- **e2e_latency_ms**: end-to-end latency from capture timestamp to final result timestamp

## Benchmark table
| date | stage | config (resolution, process_size, detector_every_n_frames, ocr_every_n_frames, buffer) | capture_fps | detector_fps | ocr_fps | e2e_latency_ms | artifacts |
|---|---|---|---:|---:|---:|---:|---|

## Notes
- Artifacts should be committed into the repo (e.g. `benchmarks/<name>/overlay.jpg`, `benchmarks/<name>/ui_state.json`).
