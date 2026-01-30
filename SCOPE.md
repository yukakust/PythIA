# TOTAL_SCOPE
Build a vision layer that understands what is on the screen in real time and outputs structured UI JSON:
- Screen capture (30 fps)
- Frame queue/buffer
- Preprocess (including optional downscale)
- UI element detection (bbox + class)
- Tracking (stable_id across frames)
- OCR (bbox/event-based + cache)
- Structure assembly (flat list and/or UI tree)
- JSON schema and stream API
- Metrics (fps/latency) + debugging

# 1st_STEP_SCOPE
Build the "Screen Capture Pipeline" module (no ML):
- Screen capture: receive screen frames in real time
- Frame buffer: keep last N frames and never block capture
- Preprocess: produce an optional ML-friendly resized frame while keeping the original
- Frame stream API: simple "get latest frame" and/or "subscribe" interface
Done criteria:
- stable ~30 fps capture
- clear timestamps
- system does not crash if downstream is slow (frames are dropped/overwritten in the buffer)
