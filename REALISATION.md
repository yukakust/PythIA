# REALISATION
 
 ## Что уже сделано
 - Захват экрана в реальном времени через `mss` (`ScreenSource`)
 - Источник кадров как абстракция `VideoSource`
   - `ScreenSource` (экран)
   - `FileSource` (видеофайл)
 - Буфер кадров `FrameBuffer` (кольцевой, с drop при перегрузе)
 - Preprocess: храним `original` и опционально `small` версию + коэффициенты масштаба
 - Демка: `examples/capture_demo.py`
 - HTTP API (FastAPI):
   - `GET /latest_frame_meta` (JSON)
   - `GET /latest_frame_jpeg` (JPEG)
   - запуск: `examples/run_api.py`
 
 ## Где срезали углы (пока нормально)
 - `FrameBuffer.frames()` выдаёт только последний новый кадр (а не все кадры подряд)
   - Это ок для realtime/дебага, но для офлайн-обработки может понадобиться другой режим
 - Нет стабильного packaging (`pyproject.toml`), демо добавляет корень репо в `sys.path`
 - Нет тестов
 - `/latest_frame_jpeg` кодирует кадр в JPEG на CPU (Pillow) — это нормально для дебага, но не для высокого FPS наружу
 
 ## Что доделать позже
 - Нормальный пакет (`pyproject.toml`) + `pip install -e .`
 - Режим буфера "выдай все кадры" для офлайн пайплайна
 - Метрики latency по кадру (time since capture)
 - API слой:
   - авторизация/лимиты
   - WebSocket/SSE для стрима меты/событий
   - более быстрый стрим кадров (например MJPEG или отдельный transport)
 - Детектор + трекер + OCR (следующие шаги)
