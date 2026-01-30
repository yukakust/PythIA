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
| 2026-01-30 | Step2.A (baseline heuristics + OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 | 23.14 | 0.64 | 0.64 | 1452.4 | benchmarks/step2A_20260130_153252/overlay_1769776376057.jpg |
| 2026-01-30 | Step2.A (baseline heuristics + OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 | 22.69 | 0.76 | 0.76 | 1128.8 | benchmarks/step2A_20260130_154012/overlay_1769776815831.jpg |
| 2026-01-30 | Step2.B (GroundingDINO + OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 box_th=0.25 text_th=0.25 device=cpu | 29.82 | 0.15 | 0.15 | 6609.3 | benchmarks/step2B_20260130_162823/overlay_1769779724564.jpg |
