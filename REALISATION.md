 # REALISATION
 
 ## What is implemented
 - Real-time screen capture via `mss` (`ScreenSource`)
 - Frame source abstraction: `VideoSource`
   - `ScreenSource` (screen)
   - `FileSource` (video file)
 - Frame buffer: `FrameBuffer` (ring buffer, drops on overload)
 - Preprocess: keep `original` and optional `small` version + scale factors
 - Demos:
   - `examples/capture_demo.py`
   - `examples/run_api.py`
 - HTTP API (FastAPI):
   - `GET /latest_frame_meta` (JSON)
   - `GET /latest_frame_jpeg` (JPEG)
 - Packaging: `pyproject.toml` + `pip install -e .`
 - Best-effort sequential stream for offline/video use: `frames_all()`
 
 ## Shortcuts (acceptable for now)
 - `FrameBuffer.frames()` yields only the latest new frame (good for realtime/debug)
 - `/latest_frame_jpeg` encodes JPEG on CPU (Pillow) — fine for debugging, not for high-FPS streaming
 - No tests
 
 ## TODO later
 - API layer:
   - auth/rate limits
   - WebSocket/SSE for streaming meta/events
   - faster video transport (e.g. MJPEG or another transport)
 - UI detection + tracking + OCR (next big steps)

 ## Vision benchmarks (Step2)
 - Step2.A: baseline heuristics UI + macOS Vision OCR
 - Step2.B: GroundingDINO UI + macOS Vision OCR
 - Step2.C: OmniParser integration via `examples/step2_omniparser.py`
   - mode=2: OmniParser UI + macOS Vision OCR
   - mode=3: OmniParser UI+content + macOS Vision OCR
   - mode=1: OmniParser UI + Omni OCR (PaddleOCR) + content

 Results are appended into `Benchmarks_along_the_way.md` and visual artifacts are saved under `benchmarks/<run-name>/`.
