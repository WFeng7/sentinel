# HLS Stream + YOLOv8 + ByteTrack

Minimal hackathon demo: read an HLS `.m3u8` stream with OpenCV, run YOLOv8 vehicle detection, track with ByteTrack, and overlay bounding boxes + track IDs in a live window.

## Install

```bash
# Optional but recommended: use a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**HLS / OpenCV:** For HLS streams, OpenCV often needs FFmpeg. Install it if the stream fails to open:

- **macOS (Homebrew):** `brew install ffmpeg`
- **Ubuntu/Debian:** `sudo apt install ffmpeg`
- **Windows:** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

On first run, the script will download the default YOLO model (`yolov8n.pt`) if missing.

## Run

Default stream and model:

```bash
python object_tracking.py
```

### Event detection + VLM analysis

When traffic changes (sustained bbox overlap, rapid deceleration), the pipeline can trigger a VLM call to produce structured incident reports.

```bash
# Enable event detection only (no VLM, no API key needed)
SENTINEL_EVENT_DETECTION=1 python object_tracking.py

# Enable event detection + VLM (requires OpenAI API key)
export OPENAI_API_KEY=sk-...
SENTINEL_EVENT_DETECTION=1 python object_tracking.py
```

**VLM output:** Machine-readable JSON (incident type, severity, actors, evidence, RAG tags) plus a human-readable dispatch note. Use it for pipelines, RAG retrieval, and playbook routing.

**Controls:** Press **`q`** in the video window to quit, **`c`** to clear selection.

## Troubleshooting

- **`ModuleNotFoundError: cv2`**  
  Install OpenCV: `pip install opencv-python`. Use the same Python you run the script with (e.g. `python3 -m pip install opencv-python`).

- **Stream never opens / black or frozen**  
  - Install FFmpeg (see Install).  
  - Confirm the URL works in VLC or a browser.  
  - The script will try to reopen the stream after 10 consecutive read failures.

- **Low FPS**  
  Use `--stride 2` or `--stride 3`, and/or `--resize_width 640` (or lower) to reduce work per frame.

- **No `python` command**  
  Use `python3` instead of `python` (e.g. `python3 main.py`).
