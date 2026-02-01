"""
Extract 3 keyframes (first, middle, last) from a video and send to VLM.
Then run RAG decision locally (backend/app/rag/data).
All local - no backend server needed.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is on sys.path so backend.app.* is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = Path(_script_dir).resolve().parents[3]
sys.path.insert(0, str(_repo_root))

# Load .env from script dir or parent
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_script_dir, ".env"))
    load_dotenv()  # also check cwd
except ImportError:
    pass  # dotenv optional

import cv2

from backend.app.motion_first.vlm import EventAnalyzer, render_human_narrative


def read_frame_at(cap: cv2.VideoCapture, idx: int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return ok, frame


def extract_keyframes(video_path: str, jpeg_quality: int = 85) -> list[tuple[float, bytes]]:
    """Extract first, middle, last frame as (ts, jpeg_bytes)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
        cap.release()
        cap = cv2.VideoCapture(video_path)
    if total <= 0:
        raise RuntimeError("Could not determine frame count.")

    indices = [0, total // 2, max(0, total - 1)]
    keyframes: list[tuple[float, bytes]] = []

    for idx in indices:
        ok, frame = read_frame_at(cap, idx)
        if not ok:
            raise RuntimeError(f"Failed to read frame at index {idx} (total={total}).")
        ts = idx / fps
        _, enc = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        )
        keyframes.append((ts, enc.tobytes()))

    cap.release()
    return keyframes


def vlm_to_rag_input(result) -> dict:
    """Build RAG /rag/decide request from VLM output."""
    ev = getattr(result, "event", None)
    rag_info = getattr(result, "rag", None)
    evidence = getattr(result, "evidence", None) or []

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

    return {
        "event_type_candidates": event_type_candidates,
        "signals": signals[:10],
        "city": "Providence",
    }


def main():
    p = argparse.ArgumentParser(
        description="Extract 3 keyframes from video, run VLM analysis, then RAG decision."
    )
    p.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to video file or HLS stream URL (required)",
    )
    p.add_argument(
        "--no-rag",
        action="store_true",
        help="Skip RAG decision (VLM only)",
    )
    args = p.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    print(f"Extracting keyframes from {args.video}...")
    keyframes = extract_keyframes(args.video)
    print(f"Got {len(keyframes)} keyframes (first, middle, last)")

    names = ["first", "middle", "last"]
    for (ts, jpeg_bytes), name in zip(keyframes, names):
        out_path = os.path.join(_script_dir, f"kf_{name}.jpg")
        with open(out_path, "wb") as f:
            f.write(jpeg_bytes)
        print(f"Wrote {out_path}")

    ctx = {
        "event_id": f"evt_video_{os.path.basename(args.video)}",
        "camera_id": "cam00",
        "fps": 30.0,
        "window_seconds": 0,
        "cv_notes": "Keyframes only; no tracking context.",
    }

    analyzer = EventAnalyzer(api_key=api_key)
    result = analyzer.analyze_from_dict(ctx, keyframes=keyframes)

    print("\n--- VLM Structured Output ---")
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    print("\n--- Human Narrative ---")
    print(render_human_narrative(result))

    if not args.no_rag:
        from backend.app.rag import create_rag_pipeline, DecisionInput

        rag_data_dir = Path(_script_dir).resolve().parent / "rag" / "data"
        _, _, engine = create_rag_pipeline(
            provider_type="local",
            data_dir=rag_data_dir if rag_data_dir.exists() else None,
        )
        inp = DecisionInput(**vlm_to_rag_input(result))
        rag_out = engine.decide(inp)
        print("\n--- RAG Input (from VLM) ---")
        print(json.dumps(inp.model_dump(), indent=2))
        print("\n--- RAG Decision (local, backend/app/rag/data) ---")
        print(json.dumps(rag_out.to_dict(), indent=2))


if __name__ == "__main__":
    main()
