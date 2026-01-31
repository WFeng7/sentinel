import html
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from ultralytics import YOLO

app = FastAPI(title="Sentinel API")

frontend_origin = os.getenv("DEPLOYMENT_FRONTEND_ORIGIN", "http://localhost:5173")

TRACKING_MODEL_PATH = os.getenv("TRACKING_MODEL_PATH", "yolo26s.pt")
TRACKING_CONF = float(os.getenv("TRACKING_CONF", "0.25"))
TRACKING_IOU = float(os.getenv("TRACKING_IOU", "0.5"))
TRACKING_IMGSZ = int(os.getenv("TRACKING_IMGSZ", "640"))
TRACKING_FPS = float(os.getenv("TRACKING_FPS", "0"))
TRACKING_MAX_BOXES = int(os.getenv("TRACKING_MAX_BOXES", "100"))
TRACKING_MAX_WIDTH = int(os.getenv("TRACKING_MAX_WIDTH", "640"))
TRACKING_MAX_HEIGHT = int(os.getenv("TRACKING_MAX_HEIGHT", "360"))
TRACKING_STRIDE = int(os.getenv("TRACKING_STRIDE", "2"))
TRACKING_TRACKER = os.getenv("TRACKING_TRACKER", "bytetrack.yaml")
TRACKING_DEVICE = os.getenv("TRACKING_DEVICE")
TRACKING_HALF = os.getenv("TRACKING_HALF", "1") == "1"
TRACKING_CLASSES = [2, 3, 5, 7]
TRACKING_ANNOTATE_FPS = float(os.getenv("TRACKING_ANNOTATE_FPS", "6"))
DEFAULT_STREAM_URL = os.getenv(
    "TRACKING_DEFAULT_STREAM_URL",
    "https://cdn3.wowza.com/1/cHVVQ21DWGJ2Qnpy/bDhTTVlD/hls/live/playlist.m3u8",
)
HLS_ROOT = Path(os.getenv("HLS_ROOT", "/tmp/sentinel-hls"))
HLS_SEGMENT_TIME = float(os.getenv("HLS_SEGMENT_TIME", "2"))
HLS_PLAYLIST_SIZE = int(os.getenv("HLS_PLAYLIST_SIZE", "6"))

tracking_lock = threading.Lock()
tracking_state = {
    "thread": None,
    "stop": threading.Event(),
    "stream_url": None,
    "labels": [],
    "boxes": [],
    "frame_size": None,
    "timestamp": 0.0,
}
tracking_model: Optional[YOLO] = None

hls_state = {
    "process": None,
    "stream_url": None,
    "output_dir": None,
    "started_at": 0.0,
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HLS_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/hls", StaticFiles(directory=str(HLS_ROOT)), name="hls")


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.on_event("startup")
async def start_default_hls():
    if hls_state.get("process"):
        return
    _stop_hls_process()
    output_dir = _start_hls(DEFAULT_STREAM_URL)
    if output_dir is None:
        print("Failed to start default HLS stream")
        return
    hls_state["output_dir"] = output_dir


DOT_CAMERAS_PAGE = "https://www.dot.ri.gov/travel/cameras_metro.php"
DOT_BASE_URL = "https://www.dot.ri.gov"
CAMERA_CACHE_TTL_SECONDS = 300
CAMERA_CACHE = {"timestamp": 0.0, "cameras": []}
CAMERA_JS_FALLBACKS = [
    "https://www.dot.ri.gov/travel/js/2cameras.js",
    "https://www.dot.ri.gov/travel/js/cameras.js",
    "https://www.dot.ri.gov/travel/2cameras.js",
    "https://www.dot.ri.gov/travel/cameras.js",
]


def extract_js_urls(html_text: str) -> list[str]:
    matches = re.findall(r"<script[^>]+src=['\"]([^'\"]+)['\"]", html_text, flags=re.I)
    urls: list[str] = []
    for src in matches:
        if src.startswith("//"):
            urls.append(f"https:{src}")
        elif src.startswith("/"):
            urls.append(f"{DOT_BASE_URL}{src}")
        elif src.startswith("http"):
            urls.append(src)
        else:
            urls.append(urljoin(DOT_CAMERAS_PAGE, src))
    return list(dict.fromkeys(urls))


def clean_camera_alt(alt: str) -> str:
    alt = (alt or "").strip()
    prefix = "Camera at "
    if alt.startswith(prefix):
        alt = alt[len(prefix) :].strip()
    return alt


def extract_labels_by_cam_id(page_html: str) -> dict[str, str]:
    cleaned = html.unescape(page_html)
    labels: dict[str, str] = {}
    pattern = re.compile(
        r"""<a\b[^>]*\bid\s*=\s*['"](?P<aid>cam\d+)['"][^>]*>.*?<img\b[^>]*\balt\s*=\s*['"](?P<alt>[^'"]*)['"][^>]*>.*?</a>""",
        flags=re.I | re.S,
    )
    for m in pattern.finditer(cleaned):
        cam_id = m.group("aid").strip()
        alt = clean_camera_alt(m.group("alt"))
        if alt:
            labels[cam_id] = alt
    return labels


def extract_id_to_m3u8_from_text(text: str) -> dict[str, str]:
    cleaned = html.unescape(text)
    id_to_url: dict[str, str] = {}
    pattern = re.compile(
        r"""document\.getElementById\(\s*['"](?P<id>cam\d+)['"]\s*\)\s*\.addEventListener\(\s*['"]click['"]\s*,\s*function\s*\(\s*\)\s*\{\s*openVideoPopup2?\s*\(\s*['"](?P<url>https?://[^'"\s]+\.m3u8(?:\?[^'"\s]+)?)['"]\s*\)\s*;?\s*\}\s*\)\s*;?""",
        flags=re.I | re.S,
    )
    for m in pattern.finditer(cleaned):
        cam_id = m.group("id").strip()
        url = m.group("url").strip()
        id_to_url[cam_id] = url
    return id_to_url


def extract_inline_scripts(page_html: str) -> list[str]:
    cleaned = html.unescape(page_html)
    return re.findall(r"<script[^>]*>(.*?)</script>", cleaned, flags=re.I | re.S)


async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url, timeout=12)
    response.raise_for_status()
    return response.text


async def fetch_cameras(force_refresh: bool = False) -> list[dict]:
    now = time.time()
    if (
        not force_refresh
        and CAMERA_CACHE["cameras"]
        and now - CAMERA_CACHE["timestamp"] < CAMERA_CACHE_TTL_SECONDS
    ):
        return CAMERA_CACHE["cameras"]

    headers = {"User-Agent": "Sentinel/1.0 (+https://localhost)"}

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            page_html = await fetch_text(client, DOT_CAMERAS_PAGE)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="Failed to reach RIDOT cameras page.") from exc

        labels = extract_labels_by_cam_id(page_html)
        id_to_url: dict[str, str] = {}

        for script in extract_inline_scripts(page_html):
            id_to_url.update(extract_id_to_m3u8_from_text(script))

        js_urls = extract_js_urls(page_html) or CAMERA_JS_FALLBACKS
        for js_url in js_urls:
            try:
                js_text = await fetch_text(client, js_url)
            except httpx.HTTPError:
                continue
            id_to_url.update(extract_id_to_m3u8_from_text(js_text))

    cameras = []
    for cam_id, stream in id_to_url.items():
        label = labels.get(cam_id, "")
        cameras.append({"id": cam_id, "label": label, "stream": stream})

    cameras.sort(key=lambda x: (x["label"] == "", x["label"].lower(), x["id"]))
    CAMERA_CACHE["timestamp"] = now
    CAMERA_CACHE["cameras"] = cameras
    return cameras


@app.get("/cameras")
async def list_cameras(limit: int = 20, refresh: bool = False):
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be at least 1")

    cameras = await fetch_cameras(force_refresh=refresh)
    sliced = cameras[:limit]
    return {
        "count": len(sliced),
        "cameras": sliced,
        "source": "dot.ri.gov",
    }


class TrackingStartRequest(BaseModel):
    stream_url: str


class TrackingStatusResponse(BaseModel):
    stream_url: Optional[str]
    frame_width: Optional[int]
    frame_height: Optional[int]
    boxes: list[list[float]]
    labels: list[str]
    timestamp: float


class HlsStartRequest(BaseModel):
    stream_url: str


def _load_tracking_model() -> YOLO:
    global tracking_model
    if tracking_model is None:
        tracking_model = YOLO(TRACKING_MODEL_PATH)
    return tracking_model


def _draw_boxes(frame, boxes, labels):
    for idx, box in enumerate(boxes):
        if not box or len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = labels[idx] if idx < len(labels) else ""
        if label:
            cv2.putText(
                frame,
                label,
                (x1 + 4, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return frame


def _tracking_loop(stream_url: str, stop_event: threading.Event):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        with tracking_lock:
            tracking_state["stream_url"] = stream_url
            tracking_state["boxes"] = []
            tracking_state["labels"] = []
            tracking_state["frame_size"] = None
            tracking_state["timestamp"] = time.time()
        return

    model = _load_tracking_model()
    fps_interval = 1.0 / TRACKING_FPS if TRACKING_FPS > 0 else 0.0
    frame_index = 0

    while not stop_event.is_set():
        start = time.time()
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        frame_height, frame_width = frame.shape[:2]
        frame_index += 1

        if TRACKING_STRIDE > 1 and frame_index % TRACKING_STRIDE != 0:
            continue

        resized_frame = frame
        if frame_width > TRACKING_MAX_WIDTH or frame_height > TRACKING_MAX_HEIGHT:
            scale = min(TRACKING_MAX_WIDTH / frame_width, TRACKING_MAX_HEIGHT / frame_height)
            resized_frame = cv2.resize(
                frame,
                (int(frame_width * scale), int(frame_height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        results = model.track(
            resized_frame,
            persist=True,
            conf=TRACKING_CONF,
            iou=TRACKING_IOU,
            imgsz=TRACKING_IMGSZ,
            classes=TRACKING_CLASSES,
            max_det=TRACKING_MAX_BOXES,
            tracker=TRACKING_TRACKER,
            verbose=False,
            device=TRACKING_DEVICE,
            half=TRACKING_HALF,
        )

        boxes_out: list[list[float]] = []
        labels_out: list[str] = []

        scale_x = frame_width / resized_frame.shape[1]
        scale_y = frame_height / resized_frame.shape[0]

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.xyxy is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.tolist()
                    boxes_out.append(
                        [
                            float(x1 * scale_x),
                            float(y1 * scale_y),
                            float(x2 * scale_x),
                            float(y2 * scale_y),
                        ]
                    )
                    if ids is not None and idx < len(ids):
                        labels_out.append(str(int(ids[idx])))
                    else:
                        labels_out.append("-")

        with tracking_lock:
            tracking_state["stream_url"] = stream_url
            tracking_state["boxes"] = boxes_out
            tracking_state["labels"] = labels_out
            tracking_state["frame_size"] = (frame_width, frame_height)
            tracking_state["timestamp"] = time.time()

        print(
            f"tracking: {len(boxes_out)} boxes on {stream_url} "
            f"({frame_width}x{frame_height})"
        )

        elapsed = time.time() - start
        sleep_time = fps_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()


@app.post("/tracking/start")
async def start_tracking(payload: TrackingStartRequest):
    if not payload.stream_url:
        raise HTTPException(status_code=400, detail="stream_url is required")

    with tracking_lock:
        current_stream = tracking_state["stream_url"]
        existing_thread = tracking_state["thread"]

    if current_stream == payload.stream_url and existing_thread and existing_thread.is_alive():
        return {"status": "already_running"}

    with tracking_lock:
        if tracking_state["thread"] and tracking_state["thread"].is_alive():
            tracking_state["stop"].set()

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_tracking_loop,
        args=(payload.stream_url, stop_event),
        daemon=True,
    )

    with tracking_lock:
        tracking_state["stop"] = stop_event
        tracking_state["thread"] = thread
        tracking_state["stream_url"] = payload.stream_url
        tracking_state["boxes"] = []
        tracking_state["labels"] = []
        tracking_state["frame_size"] = None
        tracking_state["timestamp"] = time.time()

    thread.start()
    return {"status": "started"}


@app.post("/tracking/stop")
async def stop_tracking():
    with tracking_lock:
        if tracking_state["thread"] and tracking_state["thread"].is_alive():
            tracking_state["stop"].set()
            tracking_state["thread"] = None
        tracking_state["stream_url"] = None
        tracking_state["boxes"] = []
        tracking_state["labels"] = []
        tracking_state["frame_size"] = None
        tracking_state["timestamp"] = time.time()
    return {"status": "stopped"}


@app.get("/tracking/status", response_model=TrackingStatusResponse)
async def tracking_status():
    with tracking_lock:
        frame_size = tracking_state["frame_size"]
        width = frame_size[0] if frame_size else None
        height = frame_size[1] if frame_size else None
        return TrackingStatusResponse(
            stream_url=tracking_state["stream_url"],
            frame_width=width,
            frame_height=height,
            boxes=tracking_state["boxes"],
            labels=tracking_state["labels"],
            timestamp=tracking_state["timestamp"],
        )


def _annotated_frames(stream_url: str, stop_event: threading.Event):
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        return

    model = _load_tracking_model()
    frame_index = 0
    fps_interval = 1.0 / TRACKING_ANNOTATE_FPS if TRACKING_ANNOTATE_FPS > 0 else 0.0

    while cap.isOpened() and not stop_event.is_set():
        start = time.time()
        success, frame = cap.read()
        if not success:
            time.sleep(0.05)
            continue

        frame_height, frame_width = frame.shape[:2]
        frame_index += 1

        if TRACKING_STRIDE > 1 and frame_index % TRACKING_STRIDE != 0:
            continue

        resized_frame = frame
        if frame_width > TRACKING_MAX_WIDTH or frame_height > TRACKING_MAX_HEIGHT:
            scale = min(TRACKING_MAX_WIDTH / frame_width, TRACKING_MAX_HEIGHT / frame_height)
            resized_frame = cv2.resize(
                frame,
                (int(frame_width * scale), int(frame_height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        results = model.track(
            resized_frame,
            persist=True,
            conf=TRACKING_CONF,
            iou=TRACKING_IOU,
            imgsz=TRACKING_IMGSZ,
            classes=TRACKING_CLASSES,
            max_det=TRACKING_MAX_BOXES,
            tracker=TRACKING_TRACKER,
            verbose=False,
            device=TRACKING_DEVICE,
            half=TRACKING_HALF,
        )

        boxes_out: list[list[float]] = []
        labels_out: list[str] = []

        scale_x = frame_width / resized_frame.shape[1]
        scale_y = frame_height / resized_frame.shape[0]

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and result.boxes.xyxy is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
                for idx, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.tolist()
                    boxes_out.append(
                        [
                            float(x1 * scale_x),
                            float(y1 * scale_y),
                            float(x2 * scale_x),
                            float(y2 * scale_y),
                        ]
                    )
                    if ids is not None and idx < len(ids):
                        labels_out.append(str(int(ids[idx])))
                    else:
                        labels_out.append("-")

        yield frame_width, frame_height, _draw_boxes(frame.copy(), boxes_out, labels_out)

        elapsed = time.time() - start
        sleep_time = fps_interval - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    cap.release()


@app.get("/tracking/annotated")
async def tracking_annotated(stream_url: str):
    if not stream_url:
        raise HTTPException(status_code=400, detail="stream_url is required")

    stop_event = threading.Event()

    async def frame_generator():
        for _, _, annotated in _annotated_frames(stream_url, stop_event):
            ok, buffer = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
            if not ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")


def _stop_hls_process():
    proc = hls_state.get("process")
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    hls_state["process"] = None
    hls_state["stream_url"] = None
    hls_state["output_dir"] = None
    hls_state["started_at"] = 0.0


def _start_hls_process(width: int, height: int, output_dir: Path) -> subprocess.Popen:
    output_dir.mkdir(parents=True, exist_ok=True)
    playlist_path = output_dir / "index.m3u8"

    fps = TRACKING_ANNOTATE_FPS if TRACKING_ANNOTATE_FPS > 0 else 6

    ffmpeg_cmd = [
        "ffmpeg",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-tune",
        "zerolatency",
        "-g",
        str(int(fps * 2)),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-f",
        "hls",
        "-hls_time",
        str(HLS_SEGMENT_TIME),
        "-hls_list_size",
        str(HLS_PLAYLIST_SIZE),
        "-hls_start_number_source",
        "epoch",
        "-hls_flags",
        "delete_segments+omit_endlist+independent_segments",
        "-hls_playlist_type",
        "live",
        str(playlist_path),
    ]

    return subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)


def _start_hls(stream_url: str):
    output_dir = HLS_ROOT / f"stream-{abs(hash(stream_url))}"
    stop_event = threading.Event()
    frames = _annotated_frames(stream_url, stop_event)

    width = None
    height = None
    for width, height, _ in frames:
        break
    if width is None or height is None:
        return None

    proc = _start_hls_process(width, height, output_dir)

    hls_state["process"] = proc
    hls_state["stream_url"] = stream_url
    hls_state["output_dir"] = output_dir
    hls_state["started_at"] = time.time()

    def feed():
        nonlocal proc
        for _, _, annotated in frames:
            if stop_event.is_set():
                break
            if proc.poll() is not None:
                proc = _start_hls_process(width, height, output_dir)
                hls_state["process"] = proc
            frame = np.ascontiguousarray(annotated)
            expected = width * height * 3
            if frame.size * frame.itemsize != expected:
                continue
            try:
                proc.stdin.write(frame.tobytes())
                proc.stdin.flush()
            except Exception:
                proc = _start_hls_process(width, height, output_dir)
                hls_state["process"] = proc
                continue
        stop_event.set()
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass

    thread = threading.Thread(target=feed, daemon=True)
    thread.start()

    return output_dir


@app.post("/tracking/hls/start")
async def start_tracking_hls(payload: HlsStartRequest):
    if not payload.stream_url:
        raise HTTPException(status_code=400, detail="stream_url is required")

    if hls_state.get("stream_url") == payload.stream_url and hls_state.get("process"):
        return {"status": "already_running", "playlist": f"/hls/stream-{abs(hash(payload.stream_url))}/index.m3u8"}

    _stop_hls_process()
    output_dir = _start_hls(payload.stream_url)
    if output_dir is None:
        raise HTTPException(status_code=500, detail="Failed to start HLS stream")

    playlist = f"/hls/{output_dir.name}/index.m3u8"
    return {"status": "started", "playlist": playlist}


@app.post("/tracking/hls/stop")
async def stop_tracking_hls():
    _stop_hls_process()
    return {"status": "stopped"}


@app.get("/tracking/hls/default")
async def default_hls_playlist():
    output_dir = hls_state.get("output_dir")
    if output_dir is None:
        raise HTTPException(status_code=404, detail="HLS stream not running")
    return {"playlist": f"/hls/{output_dir.name}/index.m3u8"}


@app.get("/hls/{stream_dir}/index.m3u8")
async def hls_playlist(stream_dir: str):
    playlist_path = HLS_ROOT / stream_dir / "index.m3u8"
    if not playlist_path.exists():
        raise HTTPException(status_code=404, detail="playlist not found")
    content = playlist_path.read_text()
    return Response(content, media_type="application/vnd.apple.mpegurl")
