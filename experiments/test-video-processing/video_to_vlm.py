"""
Extract 3 keyframes (first, middle, last) from a video and send to VLM.
No tracking/velocity context - images only.
"""

import argparse
import json
import os
import sys

# Allow importing vlm from parent
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_script_dir))

# Load .env from script dir or parent
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_script_dir, ".env"))
    load_dotenv()  # also check cwd
except ImportError:
    pass  # dotenv optional

import cv2

from vlm import EventAnalyzer, render_human_narrative


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


def main():
    p = argparse.ArgumentParser(
        description="Extract 3 keyframes from video and run VLM analysis."
    )
    p.add_argument(
        "video",
        nargs="?",
        default="cropped.mov",
        help="Path to video (default: cropped.mov)",
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
    print(json.dumps(result, indent=2))
    print("\n--- Human Narrative ---")
    print(render_human_narrative(result))


if __name__ == "__main__":
    main()
