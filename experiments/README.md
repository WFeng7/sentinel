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
python main.py
```

With options:

```bash
python main.py --url "https://cdn3.wowza.com/1/eUp6WUZ2Q0NiTnNh/K2VvaWJo/hls/fj9xgx7n/464/chunklist.m3u8" --model yolov8n.pt --stride 2
```

**Useful arguments:**

| Argument        | Default | Description                                      |
|----------------|--------|--------------------------------------------------|
| `--url`        | (Wowza) | HLS stream URL (.m3u8)                          |
| `--model`      | yolov8n.pt | YOLO model path or name (e.g. yolov8s.pt)   |
| `--stride`     | 1      | Run YOLO every N frames (2 or 3 for speed)       |
| `--conf`       | 0.25   | Detection confidence threshold                  |
| `--iou`        | 0.45   | NMS IOU threshold                                |
| `--imgsz`      | 640    | YOLO input size                                 |
| `--max_fps`    | (auto) | Cap display FPS                                  |
| `--resize_width` | (none) | Resize frame width for faster processing        |

**Controls:** Press **`q`** in the video window to quit.

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
