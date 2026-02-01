import asyncio
import html
import json
import os
import re
import time
from datetime import datetime, timezone
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.rag import (
    DecisionEngine,
    DecisionInput,
    create_rag_pipeline,
)
from openai import OpenAI

app = FastAPI(title="Sentinel API")

# RAG decision layer (Stage 3) - lazy init with MockPolicyProvider
_rag_engine: DecisionEngine | None = None


def get_decision_engine() -> DecisionEngine:
    global _rag_engine
    if _rag_engine is None:
        _, _, _rag_engine = create_rag_pipeline()
    return _rag_engine

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
    except OSError:
        return


def append_geo_llm_cache(entry: dict) -> None:
    try:
        with open(GEO_LLM_CACHE_FILE, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        return


load_geo_llm_cache()


async def geocode_with_llm(label: str) -> dict | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    cached = GEO_LLM_CACHE.get(label)
    if cached:
        if cached.get("miss"):
            return None
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
        response = client.responses.create(
            model="gpt-4.1-nano",
            input=prompt,
            temperature=0.1,
        )
        text = response.output_text.strip()
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
    except Exception:
        pass

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


async def enrich_cameras_with_geo(cameras: list[dict]) -> None:
    missing = [cam for cam in cameras if cam.get("label")]
    if not missing:
        return

    headers = {"User-Agent": "Sentinel/1.0 (+https://localhost)"}
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        last_request = 0.0
        for cam in missing:
            label = cam.get("label", "")
            cached = get_cached_geo(label)
            if cached:
                cam.update(cached)
                continue
            if label in GEO_CACHE and GEO_CACHE[label].get("miss"):
                continue
            llm_geo = None

            now = time.time()
            wait_for = 1.0 - (now - last_request)
            if wait_for > 0:
                await asyncio.sleep(wait_for)

            try:
                geo = await geocode_label(client, label)
            except httpx.HTTPError:
                geo = None

            last_request = time.time()
            if geo:
                cam.update(geo)
            else:
                llm_geo = await geocode_with_llm(label)
                if llm_geo:
                    cam.update(llm_geo)
                else:
                    print(f"[geocode] miss label='{label}' id='{cam.get('id','')}'")
            set_cached_geo(label, geo or llm_geo)


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
    await enrich_cameras_with_geo(sliced)
    return {
        "count": len(sliced),
        "cameras": sliced,
        "source": "dot.ri.gov",
    }


# ---------------------------------------------------------------------------
# RAG Decision (Stage 3)
# ---------------------------------------------------------------------------


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
    Placeholder VLM endpoint for control-panel location clicks.
    Accepts: { "camera_id": "...", "label": "...", "stream_url": "..." }
    Returns: light-weight analysis summary (stub if VLM not wired yet).
    """
    camera_id = body.get("camera_id") or body.get("id") or "unknown"
    label = body.get("label") or ""
    stream_url = body.get("stream_url") or body.get("url") or ""
    requested_at = datetime.now(timezone.utc).isoformat()

    # Toggleable hook for future VLM integration.
    if os.getenv("SENTINEL_VLM_MODE", "stub").lower() != "stub":
        return {
            "status": "not_configured",
            "camera_id": camera_id,
            "label": label,
            "stream_url": stream_url,
            "summary": "VLM mode is enabled but no analyzer is configured yet.",
            "observations": [
                "Attach a keyframe extraction pipeline to this endpoint.",
                "Then call the vision model with the keyframe(s).",
            ],
            "updated_at": requested_at,
        }

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
