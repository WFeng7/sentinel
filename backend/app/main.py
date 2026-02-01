import asyncio
import base64
import html
import json
import os
import re
import time
import sys
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from collections import deque
from urllib.parse import urljoin

import cv2
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.vlm import EventAnalyzer, render_human_narrative, EventContext
from app.rag import (
    DecisionEngine,
    DecisionInput,
    create_rag_pipeline,
)
from openai import OpenAI
from dotenv import load_dotenv

# Load env from backend/.env and project root .env if present.
_backend_env = os.path.join(os.path.dirname(__file__), "..", ".env")
_root_env = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
load_dotenv(dotenv_path=_backend_env, override=True)
load_dotenv(dotenv_path=_root_env, override=False)

app = FastAPI(title="Sentinel API")

_rag_engine: DecisionEngine | None = None
_incident_log: deque = deque(maxlen=100)
_camera_label_cache: dict[str, str] = {}
_camera_label_cache_ts: float = 0.0
_camera_label_cache_ttl_s = 300.0


def get_decision_engine() -> DecisionEngine:
    global _rag_engine
    if _rag_engine is None:
        _, _, _rag_engine = create_rag_pipeline()
    return _rag_engine

_vlm_analyzer: EventAnalyzer | None = None

def get_vlm_analyzer():
    global _vlm_analyzer
    if _vlm_analyzer is None:
        _vlm_analyzer = EventAnalyzer(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model="gpt-4o-mini"  # Use cheaper model
        )
    return _vlm_analyzer


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://brown-sentinel.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "ok"}


def _load_camera_label_map() -> dict[str, str]:
    global _camera_label_cache, _camera_label_cache_ts
    now = time.time()
    if _camera_label_cache and (now - _camera_label_cache_ts) < _camera_label_cache_ttl_s:
        return _camera_label_cache
    cameras = load_cameras_from_file() or CAMERA_CACHE.get("cameras") or []
    label_map: dict[str, str] = {}
    for cam in cameras:
        cam_id = cam.get("id")
        label = cam.get("label") or ""
        if cam_id:
            label_map[cam_id] = label
    _camera_label_cache = label_map
    _camera_label_cache_ts = now
    return label_map


def _resolve_camera_label(camera_id: str | None) -> str:
    if not camera_id:
        return ""
    label_map = _load_camera_label_map()
    return label_map.get(camera_id, "")


@app.get("/incidents")
async def list_incidents(limit: int = 20):
    items = list(_incident_log)
    return {"incidents": items[: max(1, min(limit, 100))]}


@app.post("/incidents")
async def add_incident(body: dict):
    camera_id = body.get("camera_id") or ""
    event_id = body.get("event_id") or f"evt_{camera_id}_{int(time.time())}"
    score = body.get("score")
    events = body.get("events")
    timestamp = body.get("timestamp") or datetime.now(timezone.utc).isoformat()
    label = body.get("label") or _resolve_camera_label(camera_id)
    incident = {
        "id": event_id,
        "camera_id": camera_id,
        "label": label,
        "score": score,
        "events": events,
        "timestamp": timestamp,
    }
    _incident_log.appendleft(incident)
    return {"status": "ok", "incident": incident}


@app.get("/fake-camera/{filename}")
async def fake_camera_file(filename: str):
    if filename != "2026-01-3015-25-54.mov":
        raise HTTPException(status_code=404, detail="Fake camera file not found")
    repo_root = Path(__file__).resolve().parents[2]
    file_path = repo_root / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Fake camera file not found")
    return FileResponse(path=str(file_path), filename=file_path.name)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@app.get("/workers/motion-first/status")
async def motion_first_status():
    running = _multi_worker_proc is not None and _multi_worker_proc.poll() is None
    return {
        "running": running,
        "pid": _multi_worker_proc.pid if running else None,
        "started_at": _multi_worker_started_at,
    }


@app.post("/workers/motion-first/start")
async def motion_first_start(
    max_workers: int | None = None,
    target_fps: float | None = None,
    threshold: float | None = None,
    window_frames: int | None = None,
    incident_rate_limit: float | None = None,
    enable_vlm: bool | None = None,
    enable_rag: bool | None = None,
    testincident: bool | None = None,
):
    global _multi_worker_proc, _multi_worker_started_at
    if _multi_worker_proc is not None and _multi_worker_proc.poll() is None:
        return {
            "status": "already_running",
            "pid": _multi_worker_proc.pid,
            "started_at": _multi_worker_started_at,
        }

    max_workers = max_workers or int(os.environ.get("SENTINEL_MULTI_MAX_WORKERS", "2"))
    target_fps = target_fps or float(os.environ.get("SENTINEL_MULTI_TARGET_FPS", "5.0"))
    threshold = threshold or float(os.environ.get("SENTINEL_MULTI_THRESHOLD", "4.0"))
    window_frames = window_frames or int(os.environ.get("SENTINEL_MULTI_WINDOW_FRAMES", "50"))
    incident_rate_limit = incident_rate_limit or float(os.environ.get("SENTINEL_MULTI_RATE_LIMIT_S", "60.0"))
    if enable_vlm is None:
        enable_vlm = _env_bool("SENTINEL_MULTI_ENABLE_VLM", False)
    if enable_rag is None:
        enable_rag = _env_bool("SENTINEL_MULTI_ENABLE_RAG", False)
    if enable_rag and not enable_vlm:
        enable_vlm = True

    if testincident is None:
        testincident = _env_bool("TESTINCIDENT", False)

    cmd = _multi_worker_cmd(
        max_workers=max_workers,
        target_fps=target_fps,
        threshold=threshold,
        window_frames=window_frames,
        incident_rate_limit=incident_rate_limit,
        enable_vlm=enable_vlm,
        enable_rag=enable_rag,
    )
    if testincident:
        cmd.append("--test-incident")
    repo_root = Path(__file__).resolve().parents[2]
    _multi_worker_proc = subprocess.Popen(cmd, cwd=str(repo_root), env=os.environ.copy())
    _multi_worker_started_at = time.time()
    return {"status": "started", "pid": _multi_worker_proc.pid, "cmd": cmd}


@app.post("/workers/motion-first/stop")
async def motion_first_stop():
    global _multi_worker_proc
    if _multi_worker_proc is None or _multi_worker_proc.poll() is not None:
        _multi_worker_proc = None
        return {"status": "not_running"}
    _multi_worker_proc.terminate()
    try:
        _multi_worker_proc.wait(timeout=3)
    except Exception:
        _multi_worker_proc.kill()
    _multi_worker_proc = None
    return {"status": "stopped"}


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

GEO_CACHE_TTL_SECONDS = 60 * 60 * 24 * 30
GEO_CACHE: dict[str, dict] = {}
GEO_LLM_CACHE_FILE = os.path.join(os.path.dirname(__file__), "geo_llm_cache.txt")
GEO_LLM_CACHE: dict[str, dict] = {}


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


async def sample_frames_from_hls(stream_url: str, times: list[float]) -> list[tuple[float, bytes]]:
    """
    Sample frames from an HLS stream at given timestamps.
    Returns list of (timestamp, jpeg_bytes).
    """
    import tempfile
    import subprocess

    # Download a short segment of the HLS stream to a temp file
    temp_dir = tempfile.mkdtemp()
    segment_path = os.path.join(temp_dir, "segment.ts")
    try:
        # Use ffmpeg to download first 15 seconds and extract frames
        cmd = [
            "ffmpeg",
            "-i", stream_url,
            "-t", "15",
            "-avoid_negative_ts", "make_zero",
            "-c", "copy",
            "-f", "mpegts",
            "-y",
            segment_path
        ]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            print(f"[ffmpeg] error: {stderr.decode()}")
            return []

        frames = []
        for t in times:
            # Extract a frame at the requested time
            out_path = os.path.join(temp_dir, f"frame_{t}.jpg")
            cmd = [
                "ffmpeg",
                "-ss", str(t),
                "-i", segment_path,
                "-frames:v", "1",
                "-q:v", "2",
                "-y",
                out_path
            ]
            proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            await proc.communicate()
            if proc.returncode == 0 and os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    frames.append((t, f.read()))
        return frames
    finally:
        # Cleanup temp files
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except:
                pass
        try:
            os.rmdir(temp_dir)
        except:
            pass


async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url, timeout=12)
    response.raise_for_status()
    return response.text


def get_cached_geo(label: str) -> dict | None:
    cached = GEO_CACHE.get(label)
    if not cached:
        return None
    if time.time() - cached.get("timestamp", 0) > GEO_CACHE_TTL_SECONDS:
        GEO_CACHE.pop(label, None)
        return None
    if cached.get("miss"):
        return {}
    return cached.get("geo") or {}


def set_cached_geo(label: str, geo: dict | None) -> None:
    GEO_CACHE[label] = {
        "timestamp": time.time(),
        "geo": geo or {},
        "miss": geo is None,
    }


def load_geo_llm_cache() -> None:
    if not os.path.exists(GEO_LLM_CACHE_FILE):
        print(f"[geocode-llm] cache file not found at {GEO_LLM_CACHE_FILE}")
        return
    try:
        with open(GEO_LLM_CACHE_FILE, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                label = entry.get("label")
                if not label:
                    continue
                GEO_LLM_CACHE[label] = entry
        print(f"[geocode-llm] loaded {len(GEO_LLM_CACHE)} cached entries")
    except OSError:
        return


def append_geo_llm_cache(entry: dict) -> None:
    try:
        os.makedirs(os.path.dirname(GEO_LLM_CACHE_FILE), exist_ok=True)
        with open(GEO_LLM_CACHE_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(f"[geocode-llm] wrote cache entry for label='{entry.get('label','')}'")
    except OSError:
        print("[geocode-llm] failed to write cache file")
        return


load_geo_llm_cache()


async def geocode_with_llm(label: str) -> dict | None:
    if not os.getenv("OPENAI_API_KEY"):
        load_dotenv(dotenv_path=_backend_env, override=True)
        load_dotenv(dotenv_path=_root_env, override=False)
    if not os.getenv("OPENAI_API_KEY"):
        print("[geocode-llm] OPENAI_API_KEY missing; skipping LLM geocode")
        return None
    cached = GEO_LLM_CACHE.get(label)
    if cached:
        if cached.get("miss"):
            print(f"[geocode-llm] cached miss label='{label}'")
            return None
        print(f"[geocode-llm] cache hit label='{label}'")
        return cached.get("geo") or None

    client = OpenAI()
    prompt = (
        "You are a geocoding assistant. Given a short traffic camera label, "
        "return JSON with latitude/longitude near Providence, Rhode Island. "
        "Only return JSON: {\"lat\": number|null, \"lng\": number|null}. "
        "If you are unsure, return nulls.\n\n"
        f"Label: {label}\n"
    )
    try:
        print(f"[geocode-llm] calling OpenAI for label='{label}'")
        response = client.responses.create(
            model="gpt-4.1-nano",
            input=prompt,
            temperature=0.1,
        )
        text = response.output_text.strip()
        print(f"[geocode-llm] OpenAI raw response: {text}")
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        data = json.loads(text)
        lat = data.get("lat")
        lng = data.get("lng")
        if isinstance(lat, (int, float)) and isinstance(lng, (int, float)):
            geo = {"lat": float(lat), "lng": float(lng)}
            entry = {"label": label, "geo": geo, "miss": False, "ts": time.time()}
            GEO_LLM_CACHE[label] = entry
            append_geo_llm_cache(entry)
            return geo
    except Exception as exc:
        print(f"[geocode-llm] OpenAI error: {exc}")

    entry = {"label": label, "geo": None, "miss": True, "ts": time.time()}
    GEO_LLM_CACHE[label] = entry
    append_geo_llm_cache(entry)
    return None


async def geocode_label(client: httpx.AsyncClient, label: str) -> dict | None:
    def normalize(raw: str) -> str:
        text = (raw or "").strip()
        text = re.sub(r"^\d{1,3}-\d{1,3}\s+", "", text)
        text = re.sub(r"\bDMS and Camera\b", "", text, flags=re.I)
        text = re.sub(r"\bDMS\b", "", text, flags=re.I)
        text = re.sub(r"\s+", " ", text)
        text = text.replace("I-", "Interstate ").replace("I ", "Interstate ")
        text = re.sub(r"\bNB\b", "Northbound", text)
        text = re.sub(r"\bSB\b", "Southbound", text)
        text = re.sub(r"\bEB\b", "Eastbound", text)
        text = re.sub(r"\bWB\b", "Westbound", text)
        text = re.sub(r"\bRT\b", "Route", text, flags=re.I)
        text = text.replace("@", "at")
        text = re.sub(r"\s+at\s+", " at ", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    base = label.strip()
    expanded = normalize(label)
    intersection = re.sub(r"\s+at\s+", " and ", expanded, flags=re.I)
    queries = [
        f"{base}, Providence, RI",
        f"{expanded}, Providence, RI",
        f"{intersection}, Providence, RI",
        f"{base}, Rhode Island",
        f"{expanded}, Rhode Island",
        f"{intersection}, Rhode Island",
    ]
    for query in queries:
        response = await client.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "json", "limit": 1, "addressdetails": 0},
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        if not data:
            continue
        result = data[0]
        try:
            return {"lat": float(result["lat"]), "lng": float(result["lon"])}
        except (KeyError, ValueError, TypeError):
            continue
    return None


async def geocode_one(client, label):
    cached = get_cached_geo(label)
    if cached:
        print(f"[geocode] cache hit: {label}")
        return cached
    if label in GEO_CACHE and GEO_CACHE[label].get("miss"):
        print(f"[geocode] cached miss: {label}")
        return {}
    print(f"[geocode] fetching: {label}")
    try:
        geo = await geocode_label(client, label)
    except httpx.HTTPError:
        geo = None
    if not geo:
        geo = await geocode_with_llm(label)
    set_cached_geo(label, geo)
    print(f"[geocode] result: {label} -> {geo}")
    return geo or {}

async def enrich_cameras_with_geo(cameras: list[dict]) -> None:
    missing = [cam for cam in cameras if cam.get("label")]
    if not missing:
        return

    headers = {"User-Agent": "Sentinel/1.0 (+https://localhost)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        tasks = [geocode_one(client, cam["label"]) for cam in missing]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for cam, result in zip(missing, results):
            if isinstance(result, dict) and result:
                cam.update(result)


CAMERA_CACHE_FILE = os.path.join(os.path.dirname(__file__), "..", "cameras_cache.txt")

# ---------------------------------------------------------------------------
# Motion-first multi-camera worker control
# ---------------------------------------------------------------------------

_multi_worker_proc: subprocess.Popen | None = None
_multi_worker_started_at: float | None = None


def _multi_worker_cmd(
    *,
    max_workers: int,
    target_fps: float,
    threshold: float,
    window_frames: int,
    incident_rate_limit: float,
    enable_vlm: bool,
    enable_rag: bool,
) -> list[str]:
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "experiments" / "motion_first_multi.py"
    cmd = [
        sys.executable,
        str(script),
        "--no-display",
        "--max-workers",
        str(max_workers),
        "--target-fps",
        str(target_fps),
        "--threshold",
        str(threshold),
        "--window-frames",
        str(window_frames),
        "--incident-rate-limit",
        str(incident_rate_limit),
    ]
    if enable_vlm:
        cmd.append("--enable-vlm")
    if enable_rag:
        cmd.append("--enable-rag")
    return cmd

def load_cameras_from_file() -> list[dict] | None:
    """Load cameras from cache file if it exists and is recent."""
    try:
        if not os.path.exists(CAMERA_CACHE_FILE):
            return None
        
        # Check if file is less than 1 hour old
        file_age = time.time() - os.path.getmtime(CAMERA_CACHE_FILE)
        if file_age > 3600:  # 1 hour TTL
            return None
        
        with open(CAMERA_CACHE_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            if content.strip():
                return json.loads(content)
    except Exception as e:
        print(f"[cache] Error loading cameras from file: {e}")
    return None

def save_cameras_to_file(cameras: list[dict]) -> None:
    """Save cameras to cache file."""
    try:
        with open(CAMERA_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cameras, f, indent=2)
        print(f"[cache] Saved {len(cameras)} cameras to cache file")
    except Exception as e:
        print(f"[cache] Error saving cameras to file: {e}")

async def fetch_cameras(force_refresh: bool = False) -> list[dict]:
    # Try file cache first
    if not force_refresh:
        cached_cameras = load_cameras_from_file()
        if cached_cameras:
            print(f"[cache] Loaded {len(cached_cameras)} cameras from file cache")
            # Enrich with geo coordinates if missing
            await enrich_cameras_with_geo(cached_cameras)
            return cached_cameras
    
    now = time.time()
    if (
        not force_refresh
        and CAMERA_CACHE["cameras"]
        and now - CAMERA_CACHE["timestamp"] < CAMERA_CACHE_TTL_SECONDS
    ):
        cameras = CAMERA_CACHE["cameras"]
        await enrich_cameras_with_geo(cameras)
        return cameras

    print("[fetch] Fetching cameras from RIDOT...")
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
        js_tasks = [fetch_text(client, js_url) for js_url in js_urls]
        js_results = await asyncio.gather(*js_tasks, return_exceptions=True)
        for result in js_results:
            if isinstance(result, Exception):
                continue
            id_to_url.update(extract_id_to_m3u8_from_text(result))

    cameras = []
    for cam_id, stream in id_to_url.items():
        label = labels.get(cam_id, "")
        camera = {"id": cam_id, "label": label, "stream": stream}
        cameras.append(camera)
        print(f"[camera] {camera['id']} | label='{camera['label']}' | stream={camera['stream']}")

    cameras.sort(key=lambda x: (x["label"] == "", x["label"].lower(), x["id"]))
    
    # Enrich with geo coordinates
    await enrich_cameras_with_geo(cameras)
    
    # Update both caches
    CAMERA_CACHE["timestamp"] = now
    CAMERA_CACHE["cameras"] = cameras
    save_cameras_to_file(cameras)
    
    print(f"[fetch] Successfully fetched and cached {len(cameras)} cameras")
    return cameras


@app.get("/cameras")
async def list_cameras(limit: int = 20, refresh: bool = False):
    print("Called!")
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be at least 1")

    cameras = await fetch_cameras(force_refresh=refresh)
    sliced = cameras[:limit]
    return {
        "count": len(sliced),
        "cameras": sliced,
        "source": "dot.ri.gov",
    }


# ---------------------------------------------------------------------------
# VLM Event Analysis
# ---------------------------------------------------------------------------

@app.post("/vlm/analyze")
async def vlm_analyze(body: dict):
    """
    VLM event analysis endpoint.
    Expects: { "context": {...}, "keyframes": [{"ts": 0, "base64": "..."}, ...] }
    Returns: structured event output + human narrative.
    """
    context = body.get("context") or {}
    keyframes_raw = body.get("keyframes") or []
    keyframes: list[tuple[float, bytes]] = []
    for kf in keyframes_raw[:6]:
        ts = float(kf.get("ts", 0))
        b64 = kf.get("base64") or kf.get("data") or ""
        if b64:
            keyframes.append((ts, base64.standard_b64decode(b64)))
    analyzer = get_vlm_analyzer()
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    try:
        result = analyzer.analyze_from_dict(context, keyframes=keyframes if keyframes else None)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # FastAPI will automatically serialize the Pydantic model to JSON
    return {
        "result": result,
        "narrative": render_human_narrative(result),
    }

# ---------------------------------------------------------------------------
# RAG Decision (Stage 3)
# ---------------------------------------------------------------------------


@app.post("/vlm+rag")
async def vlm_and_rag(body: dict):
    """
    Combined VLM + RAG endpoint.
    Accepts: { "camera_id": "...", "label": "...", "stream_url": "..." }
    Backend samples 3 frames from the HLS URL internally.
    Returns: { vlm: {...}, rag: {...} }
    """
    camera_id = body.get("camera_id")
    label = body.get("label")
    stream_url = body.get("stream_url")
    if not (camera_id and label and stream_url):
        raise HTTPException(status_code=400, detail="camera_id, label, and stream_url required")

    # Sample 3 frames from the HLS stream on the backend
    keyframes = await sample_frames_from_hls(stream_url, times=[0, 5, 10])
    if not keyframes:
        raise HTTPException(status_code=500, detail="Failed to sample frames from stream")

    # VLM step
    ctx = EventContext(
        event_id=f"evt_{camera_id}_{int(time.time())}",
        camera_id=camera_id,
        fps=30.0,
        window_seconds=0,
        cv_notes="Auto-sampled 3 frames (0s, 5s, 10s) from HLS stream."
    )
    analyzer = get_vlm_analyzer()
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY not configured")
    try:
        vlm_result = analyzer.analyze(ctx, keyframes=keyframes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Map VLM output to RAG input (mirroring video_to_vlm.py)
    ev = getattr(vlm_result, "event", None)
    rag_info = getattr(vlm_result, "rag", None)
    evidence = getattr(vlm_result, "evidence", None) or []

    event_type_candidates = []
    if ev:
        if getattr(ev, "type", None):
            event_type_candidates.append(str(ev.type))
        if getattr(ev, "category", None):
            cat = ev.category
            event_type_candidates.append(cat.value if hasattr(cat, "value") else str(cat))
    tags = getattr(rag_info, "tags", []) if rag_info else []
    event_type_candidates.extend(tags[:5])

    signals = []
    for e in evidence:
        if isinstance(e, dict):
            if e.get("claim"):
                signals.append(str(e["claim"])[:200])
            signals.extend(e.get("signals", [])[:3])
        elif hasattr(e, "claim"):
            signals.append(str(e.claim)[:200])
    queries = getattr(rag_info, "queries", []) if rag_info else []
    signals.extend(queries[:3])

    rag_input = {
        "event_type_candidates": event_type_candidates,
        "signals": signals[:10],
        "city": "Providence",
    }

    # RAG step
    inp = DecisionInput(
        event_type_candidates=rag_input["event_type_candidates"],
        signals=rag_input["signals"],
        city=rag_input["city"],
    )
    engine = get_decision_engine()
    rag_output = engine.decide(inp)

    return {
        "vlm": {
            "result": vlm_result.model_dump(mode="json"),
            "narrative": render_human_narrative(vlm_result),
        },
        "rag": rag_output.to_dict(),
    }


@app.post("/rag/decide")
async def rag_decide(body: dict):
    """
    RAG decision endpoint.
    Expects: { "event_type_candidates": [...], "signals": [...], "city": "Providence" }
    Returns: structured JSON decision, human-readable explanation, supporting policy excerpts.
    """
    inp = DecisionInput(
        event_type_candidates=body.get("event_type_candidates") or [],
        signals=body.get("signals") or [],
        city=body.get("city") or "Providence",
    )
    engine = get_decision_engine()
    output = engine.decide(inp)
    return output.to_dict()


# ---------------------------------------------------------------------------
# VLM (placeholder) - Location insights
# ---------------------------------------------------------------------------


@app.post("/vlm/location")
async def vlm_location(body: dict):
    """
    VLM endpoint for control-panel location clicks.
    Accepts: { "camera_id": "...", "label": "...", "stream_url": "..." }
    Returns: VLM analysis of the current camera view.
    """
    camera_id = body.get("camera_id") or body.get("id") or "unknown"
    label = body.get("label") or ""
    stream_url = body.get("stream_url") or body.get("url") or ""
    requested_at = datetime.now(timezone.utc).isoformat()

    # Check if VLM mode is enabled
    if os.getenv("SENTINEL_VLM_MODE", "stub").lower() == "stub":
        return {
            "status": "stub",
            "camera_id": camera_id,
            "label": label,
            "stream_url": stream_url,
            "summary": "VLM not configured yet. This is a placeholder response.",
            "observations": [
                "No keyframe payload was provided.",
                "Wire a keyframe extractor and vision model to enrich this response.",
            ],
            "updated_at": requested_at,
        }

    try:
        # Get VLM analyzer
        analyzer = get_vlm_analyzer()
        
        # Create mock event context for location-based analysis
        from app.vlm import EventContext
        ctx = EventContext(
            event_id=f"loc_{camera_id}_{int(time.time())}",
            camera_id=camera_id,
            fps=30.0,
            window_seconds=10.0,
            cv_notes=f"Manual analysis request for camera: {label or camera_id}"
        )
        
        # Perform VLM analysis (without keyframes for now)
        result = analyzer.analyze(ctx)
        
        # Convert to frontend-friendly format
        narrative = render_human_narrative(result)
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "label": label,
            "stream_url": stream_url,
            "summary": narrative,
            "observations": [
                f"Event type: {result.event.type}",
                f"Severity: {result.event.severity.value}",
                f"Confidence: {result.event.confidence:.2f}",
                f"Actors detected: {len(result.actors)}",
            ],
            "detailed_analysis": result.model_dump(),
            "updated_at": requested_at,
        }
        
    except Exception as e:
        return {
            "status": "error",
            "camera_id": camera_id,
            "label": label,
            "stream_url": stream_url,
            "summary": f"VLM analysis failed: {str(e)}",
            "observations": ["Error occurred during analysis"],
            "updated_at": requested_at,
        }
