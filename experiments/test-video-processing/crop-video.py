import argparse
import os
import cv2


def crop_video(
    input_path: str,
    output_path: str,
    x: int,
    y: int,
    w: int,
    h: int,
    trim_end_seconds: float = 0,
) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Clamp crop rect to video bounds
    x = max(0, min(x, src_w - 1))
    y = max(0, min(y, src_h - 1))
    w = max(1, min(w, src_w - x))
    h = max(1, min(h, src_h - y))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total_frames += 1
        cap.release()
        cap = cv2.VideoCapture(input_path)
    total_duration = total_frames / fps
    frames_to_keep = total_frames
    if trim_end_seconds > 0:
        new_duration = max(0, total_duration - trim_end_seconds)
        frames_to_keep = int(new_duration * fps)
        frames_to_keep = max(0, min(frames_to_keep, total_frames))
        print(f"Trim: keeping first {frames_to_keep} frames (cutting last {trim_end_seconds}s)")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # output container is .mp4 by default
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    frame_count = 0
    while frame_count < frames_to_keep:
        ok, frame = cap.read()
        if not ok:
            break
        cropped = frame[y:y + h, x:x + w]
        out.write(cropped)
        frame_count += 1

    cap.release()
    out.release()

    if frame_count == 0:
        raise RuntimeError("No frames were written (input unreadable or empty).")

    print(f"Wrote {frame_count} frames to: {output_path}")
    print(f"Crop rect: x={x}, y={y}, w={w}, h={h}")


def main():
    p = argparse.ArgumentParser(description="Crop a video to a rectangle.")
    p.add_argument("input", help="Path to input .mov (or any video)")
    p.add_argument("output", help="Path to output video (recommended .mp4)")
    p.add_argument("--x", type=int, required=True, help="Left coordinate of crop")
    p.add_argument("--y", type=int, required=True, help="Top coordinate of crop")
    p.add_argument("--w", type=int, required=True, help="Width of crop")
    p.add_argument("--h", type=int, required=True, help="Height of crop")
    p.add_argument("--trim-end", type=float, default=0, help="Cut last N seconds from end (e.g. 5)")
    args = p.parse_args()

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    crop_video(args.input, args.output, args.x, args.y, args.w, args.h, args.trim_end)


if __name__ == "__main__":
    main()
