"""
Multi-camera runner for motion_first_tracking.py.

Spawns one worker per camera with a concurrency limit to avoid overloading
local machines.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from subprocess import Popen


def _load_cameras(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("cameras_cache.txt must be a JSON list")
    return data


def _fake_camera(api_url: str) -> dict:
    base = api_url.rstrip("/")
    return {
        "id": "fake-2026-01-3015-25-54",
        "label": "Fake Camera (2026-01-3015-25-54.mov)",
        "stream": f"{base}/fake-camera/2026-01-3015-25-54.mov",
        "lat": 41.823094,
        "lng": -71.413391,
    }


def _build_cmd(args, cam) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "motion_first_tracking.py"),
        cam["stream"],
        "--camera-id",
        cam["id"],
        "--target-fps",
        str(args.target_fps),
        "--threshold",
        str(args.threshold),
        "--window-frames",
        str(args.window_frames),
        "--incident-rate-limit",
        str(args.incident_rate_limit),
        "--device",
        args.device,
    ]
    if args.no_display:
        cmd.append("--no-display")
    if args.enable_vlm:
        cmd.append("--enable-vlm")
    if args.enable_rag:
        cmd.append("--enable-rag")
    if args.s3_bucket:
        cmd.extend(["--s3-bucket", args.s3_bucket])
    if args.s3_prefix:
        cmd.extend(["--s3-prefix", args.s3_prefix])
    if args.vlm_base_url:
        cmd.extend(["--vlm-base-url", args.vlm_base_url])
    if args.yolo:
        cmd.extend(["--yolo", args.yolo])
    if args.roi_y_frac is not None:
        cmd.extend(["--roi-y-frac", str(args.roi_y_frac)])
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Spawn motion_first_tracking.py for each camera with a worker limit."
    )
    parser.add_argument(
        "--cameras",
        default=str(Path(__file__).resolve().parents[1] / "backend" / "cameras_cache.txt"),
        help="Path to cameras_cache.txt (JSON list)",
    )
    parser.add_argument("--max-workers", type=int, default=4, help="Max concurrent workers")
    parser.add_argument("--spawn-delay", type=float, default=0.5, help="Delay between spawns (s)")
    parser.add_argument("--camera-ids", nargs="*", help="Optional list of camera IDs to run")
    parser.add_argument("--no-display", action="store_true", help="Disable OpenCV display")
    parser.add_argument("--enable-vlm", action="store_true", help="Enable VLM analysis")
    parser.add_argument("--enable-rag", action="store_true", help="Enable RAG decision")
    parser.add_argument("--s3-bucket", default=os.environ.get("SENTINEL_S3_BUCKET"))
    parser.add_argument("--s3-prefix", default=os.environ.get("SENTINEL_S3_PREFIX", "sentinel/incidents"))
    parser.add_argument("--vlm-base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--threshold", type=float, default=4.0)
    parser.add_argument("--window-frames", type=int, default=50)
    parser.add_argument("--incident-rate-limit", type=float, default=60.0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--yolo", default="yolo26s.pt")
    parser.add_argument("--roi-y-frac", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--test-incident", action="store_true", help="Only run the fake camera")
    args = parser.parse_args()

    if args.enable_rag and not args.enable_vlm:
        print("[multi] --enable-rag implies --enable-vlm; enabling VLM.")
        args.enable_vlm = True

    cameras_path = Path(args.cameras)
    cameras = _load_cameras(cameras_path)

    test_incident = args.test_incident or os.environ.get("TESTINCIDENT", "").lower() in {"1", "true", "yes", "on"}
    if test_incident:
        api_url = os.environ.get("SENTINEL_API_URL", "http://localhost:8000")
        cameras = [_fake_camera(api_url)]
    if args.camera_ids:
        allowed = set(args.camera_ids)
        cameras = [c for c in cameras if c.get("id") in allowed]

    if not cameras:
        print("[multi] No cameras matched.")
        return 1

    print(f"[multi] Cameras: {len(cameras)} | max_workers={args.max_workers}")
    if args.dry_run:
        for cam in cameras:
            print(" ".join(_build_cmd(args, cam)))
        return 0

    running: dict[int, tuple[Popen, dict]] = {}
    queue = list(cameras)

    try:
        while queue or running:
            # Fill available slots
            while queue and len(running) < args.max_workers:
                cam = queue.pop(0)
                cmd = _build_cmd(args, cam)
                print(f"[multi] starting {cam.get('id')} {cam.get('label', '')}")
                proc = Popen(cmd)
                running[proc.pid] = (proc, cam)
                time.sleep(args.spawn_delay)

            # Reap finished
            finished = []
            for pid, (proc, cam) in running.items():
                if proc.poll() is not None:
                    print(f"[multi] worker exited {cam.get('id')} pid={pid} code={proc.returncode}")
                    finished.append(pid)
            for pid in finished:
                running.pop(pid, None)

            time.sleep(0.5)
    except KeyboardInterrupt:
        print("[multi] shutting down workers...")
        for proc, _cam in running.values():
            try:
                proc.terminate()
            except Exception:
                pass
        time.sleep(1.0)
        for proc, _cam in running.values():
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
        return 130

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
