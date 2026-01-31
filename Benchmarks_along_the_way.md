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
| 2026-01-31 | Step2.C (OmniParser UI + macOS OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=2 box_th=0.05 iou_th=0.7 imgsz=640 | 18.28 | 0.16 | 0.16 | 5960.7 | benchmarks/omniparser_mode2/overlay_1769849267000.jpg |
| 2026-01-31 | Step2.C (OmniParser UI + macOS OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=2 box_th=0.05 iou_th=0.7 imgsz=640 | 20.15 | 0.15 | 0.15 | 6282.7 | benchmarks/omniparser_mode2_desktop/overlay_1769849656528.jpg |
| 2026-01-31 | Step2.C (OmniParser UI + macOS OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=2 box_th=0.05 iou_th=0.7 imgsz=640 | 11.78 | 0.07 | 0.07 | 14253.7 | benchmarks/omniparser_mode2_retry/overlay_1769850987651.jpg |
| 2026-01-31 | Step2.C (OmniParser UI + macOS OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=2 box_th=0.05 iou_th=0.7 imgsz=640 | 20.00 | 0.17 | 0.17 | 5622.2 | benchmarks/omniparser_mode2_fixui/overlay_1769851347277.jpg |
| 2026-01-31 | Step2.C (OmniParser UI + macOS OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=2 box_th=0.05 iou_th=0.7 imgsz=640 | 11.03 | 0.06 | 0.06 | 16046.0 | benchmarks/omniparser_mode2_desktop_fixui2/overlay_1769851746080.jpg |
| 2026-01-31 | Step2.C (OmniParser UI+content + macOS OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=3 box_th=0.05 iou_th=0.7 imgsz=640 | 29.81 | 0.01 | 0.01 | 93777.5 | benchmarks/omniparser_mode3_desktop_mps_bs8/overlay_1769887792070.jpg |
| 2026-01-31 | Step2.C (OmniParser UI+content + macOS OCR) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=3 box_th=0.05 iou_th=0.7 imgsz=640 | 29.96 | 0.01 | 0.01 | 122347.9 | benchmarks/omniparser_mode3_oneframe/overlay_1769888138444.jpg |
| 2026-01-31 | Step2.C (OmniParser) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=1 box_th=0.05 iou_th=0.7 imgsz=640 | 20.74 | 0.00 | 0.00 | 439292.6 | benchmarks/omniparser_mode1_oneframe/overlay_1769891241734.jpg |
| 2026-01-31 | Step2.C (OmniParser) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=1 box_th=0.05 iou_th=0.7 imgsz=640 | 4.71 | 0.02 | 0.02 | 53341.2 | benchmarks/omniparser_mode1_oneframe_ocrfix/overlay_1769891523015.jpg |
| 2026-01-31 | Step2.C (OmniParser) | resolution=orig process_size=orig detector_every_n_frames=10 ocr_every_n_frames=10 buffer=8 mode=1 box_th=0.05 iou_th=0.7 imgsz=640 | 20.29 | 0.00 | 0.00 | 355593.2 | benchmarks/omniparser_mode1_oneframe_ocrfix2/overlay_1769892525549.jpg |
