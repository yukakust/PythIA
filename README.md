# PythIA
PythIA — a real-time visual perception engine that turns raw screen pixels into structured UI understanding for AI agents.

## Step2.C (OmniParser) quick run

Artifacts are saved into `benchmarks/<run-name>/`:
- `ui_state_*.json`
- `overlay_ui_*.jpg` (UI only)
- `overlay_ocr_*.jpg` (OCR only)
- `overlay_*.jpg` (combined)

### Requirements

OmniParser (mode=1/2/3) requires an OmniParser checkout and its Python environment.

### Run (single frame)

Use `--max-frames 1` to capture/process exactly one frame.

```bash
python -m examples.step2_omniparser \
  --omni-path /ABS/PATH/TO/OmniParser \
  --mode 2 \
  --seconds 2 \
  --max-frames 1 \
  --run-name omniparser_mode2_oneframe
```

Modes:
- `--mode 2`: OmniParser UI detector + macOS Vision OCR
- `--mode 3`: OmniParser UI + local content (Florence-2) + macOS Vision OCR
- `--mode 1`: OmniParser UI + Omni OCR (PaddleOCR) + local content

### Example (mode=1)

```bash
python -m examples.step2_omniparser \
  --omni-path /ABS/PATH/TO/OmniParser \
  --mode 1 \
  --seconds 2 \
  --max-frames 1 \
  --run-name omniparser_mode1_oneframe
```
