import base64
import html
import os
import re
import time
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.vlm import EventAnalyzer, render_human_narrative
from app.rag import (
    DecisionEngine,
    DecisionInput,
    create_rag_pipeline,
)

app = FastAPI(title="Sentinel API")

# RAG decision layer (Stage 3) - lazy init with MockPolicyProvider
_rag_engine: DecisionEngine | None = None


def get_decision_engine() -> DecisionEngine:
    global _rag_engine
    if _rag_engine is None:
        _, _, _rag_engine = create_rag_pipeline()
    return _rag_engine

_vlm_analyzer: EventAnalyzer | None = None

def get_vlm_analyzer():
    global _vlm_analyzer
    if _vlm_analyzer is None:
        _vlm_analyzer = EventAnalyzer(api_key=os.environ.get("OPENAI_API_KEY"))
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
