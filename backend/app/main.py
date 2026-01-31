import html
import re
import time
from urllib.parse import urljoin

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Sentinel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
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
CAMERA_CACHE = {"timestamp": 0.0, "streams": []}
CAMERA_JS_FALLBACKS = [
    "https://www.dot.ri.gov/travel/js/2cameras.js",
    "https://www.dot.ri.gov/travel/js/cameras.js",
    "https://www.dot.ri.gov/travel/2cameras.js",
    "https://www.dot.ri.gov/travel/cameras.js",
]


def extract_js_urls(html_text: str) -> list[str]:
    matches = re.findall(r"<script[^>]+src=['\"]([^'\"]+)['\"]", html_text, flags=re.I)
    urls = []
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


def extract_m3u8_like_urls(text: str, base_url: str) -> set[str]:
    found = set()
    found |= set(re.findall(r"https?://[^\s'\"<>]+\.m3u8(?:\?[^\s'\"<>]+)?", text, flags=re.I))
    for u in re.findall(r"//[^\s'\"<>]+\.m3u8(?:\?[^\s'\"<>]+)?", text, flags=re.I):
        found.add("https:" + u)
    for u in re.findall(r"/[^\s'\"<>]+\.m3u8(?:\?[^\s'\"<>]+)?", text, flags=re.I):
        found.add(urljoin(base_url, u))
    return found


def extract_popup_urls(text: str) -> set[str]:
    cleaned = html.unescape(text)
    pattern = re.compile(
        r"openVideoPopup2?\s*\(\s*['\"](https?://[^'\"\s]+\.m3u8(?:\?[^'\"\s]+)?)['\"]\s*\)",
        flags=re.I,
    )
    matches = set(pattern.findall(cleaned))
    script_blocks = re.findall(r"<script[^>]*>(.*?)</script>", cleaned, flags=re.I | re.S)
    for script in script_blocks:
        matches.update(pattern.findall(script))
    return matches


async def fetch_text(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url, timeout=12)
    response.raise_for_status()
    return response.text


async def fetch_camera_streams(force_refresh: bool = False) -> list[str]:
    now = time.time()
    if (
        not force_refresh
        and CAMERA_CACHE["streams"]
        and now - CAMERA_CACHE["timestamp"] < CAMERA_CACHE_TTL_SECONDS
    ):
        return CAMERA_CACHE["streams"]

    headers = {"User-Agent": "Sentinel/1.0 (+https://localhost)"}
    streams = set()

    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        try:
            page_html = await fetch_text(client, DOT_CAMERAS_PAGE)
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail="Failed to reach RIDOT cameras page.") from exc

        streams |= extract_m3u8_like_urls(page_html, DOT_BASE_URL)
        streams |= extract_popup_urls(page_html)

        js_urls = extract_js_urls(page_html) or CAMERA_JS_FALLBACKS
        for js_url in js_urls:
            try:
                js_text = await fetch_text(client, js_url)
            except httpx.HTTPError:
                continue
            streams |= extract_m3u8_like_urls(js_text, DOT_BASE_URL)
            streams |= extract_popup_urls(js_text)

    stream_list = sorted(streams)
    CAMERA_CACHE["timestamp"] = now
    CAMERA_CACHE["streams"] = stream_list
    return stream_list


@app.get("/cameras")
async def list_cameras(limit: int = 20, refresh: bool = False):
    if limit < 1:
        raise HTTPException(status_code=400, detail="limit must be at least 1")

    streams = await fetch_camera_streams(force_refresh=refresh)
    return {
        "count": min(limit, len(streams)),
        "streams": streams[:limit],
        "source": "dot.ri.gov",
    }
