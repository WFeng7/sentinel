"""
Motion-first candidate scoring pipeline.

Design: Motion blobs + Kalman tracking (primary) with sparse YOLO for person detection.
Stage 1 outputs event candidates + evidence; binary "send to VLM" threshold is optional.

- Motion blobs: MOG2 on chroma (Lab a,b) → morphology → connected components → bboxes
- SORT-like tracker: Kalman (cx, cy, w, h) with dt-scaled Q + IoU matching
- Kinematics: vel, accel per track; decel spike + stopped-in-lane persistence
- Scene shockwave: median_speed, track_count, median_speed_drop, density_increase
- YOLO: sparse (1 Hz) or on pre-threshold; predict() not track()
- Frame skipping: process every skip frames at target FPS
"""

import cv2
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


# -----------------------------------------------------------------------------
# Kalman tracker (cx, cy, w, h)
# -----------------------------------------------------------------------------


@dataclass
class KalmanBox:
    """Simple Kalman filter for (cx, cy, w, h)."""

    # State: [cx, cy, w, h, vx, vy]
    # dt = 1 frame; we predict position, measure box
    x: np.ndarray  # 6x1 state
    P: np.ndarray  # 6x6 covariance

    def __post_init__(self):
        if self.x is None:
            self.x = np.zeros((6, 1))
        if self.P is None:
            self.P = np.eye(6) * 100

    @staticmethod
    def from_bbox(x1: float, y1: float, x2: float, y2: float) -> "KalmanBox":
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        x = np.array([[cx], [cy], [w], [h], [0], [0]])
        P = np.diag([10.0, 10.0, 10.0, 10.0, 100.0, 100.0])
        return KalmanBox(x=x, P=P)

    def predict(self, dt: float = 1.0) -> np.ndarray:
        F = np.eye(6)
        F[0, 4] = dt
        F[1, 5] = dt
        # Scale Q with dt: position noise ∝ dt², velocity noise ∝ dt; w/h stable
        q_pos = 1.0 * (dt**2)
        q_vel = 5.0 * dt
        Q = np.diag([q_pos, q_pos, 0.1, 0.1, q_vel, q_vel])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:4].flatten()

    def update(self, z: np.ndarray) -> np.ndarray:
        H = np.zeros((4, 6))
        H[:4, :4] = np.eye(4)
        # Bigger R on w/h to avoid noise from merges
        R = np.diag([5.0, 5.0, 8.0, 8.0])
        y = z.reshape(4, 1) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P
        return self.x[:4].flatten()

    def to_bbox(self) -> tuple[float, float, float, float]:
        cx, cy, w, h = self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    @property
    def velocity(self) -> tuple[float, float]:
        return float(self.x[4, 0]), float(self.x[5, 0])


@dataclass
class Track:
    """Single tracked motion blob."""

    track_id: int
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2
    kalman: KalmanBox
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    # Kinematics: vel, accel (rolling avg ~20 frames)
    vel_history: deque = field(default_factory=lambda: deque(maxlen=20))
    accel_history: deque = field(default_factory=lambda: deque(maxlen=20))
    vel_ema: float = 0.0
    accel_ema: float = 0.0
    alpha_ema: float = 0.095  # ~20 frame EMA span (2/(N+1))

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def area(self) -> float:
        w = self.bbox[2] - self.bbox[0]
        h = self.bbox[3] - self.bbox[1]
        return w * h

    def update_kinematics(self, dt: float, vx: float, vy: float) -> None:
        speed = math.sqrt(vx * vx + vy * vy)
        self.vel_history.append(speed)
        self.vel_ema = self.alpha_ema * speed + (1 - self.alpha_ema) * self.vel_ema
        if len(self.vel_history) >= 2 and dt > 0:
            accel = (speed - self.vel_history[-2]) / dt
            self.accel_history.append(accel)
            self.accel_ema = self.alpha_ema * accel + (1 - self.alpha_ema) * self.accel_ema

    @property
    def speed_px_s(self) -> float:
        return self.vel_ema if self.vel_history else 0.0

    @property
    def accel_px_s2(self) -> float:
        return self.accel_ema if self.accel_history else 0.0


def iou_box(box1: tuple[float, ...], box2: tuple[float, ...]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    a2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0


def iou_cost_matrix(tracks: list[Track], dets: list[tuple[float, float, float, float]]) -> np.ndarray:
    n, m = len(tracks), len(dets)
    cost = np.ones((n, m))
    for i, t in enumerate(tracks):
        for j, d in enumerate(dets):
            cost[i, j] = 1 - iou_box(t.bbox, d)
    return cost


# -----------------------------------------------------------------------------
# ROI mask
# -----------------------------------------------------------------------------


def make_road_roi_mask(h: int, w: int, y_frac: float = 0.25, bottom_frac: float = 1.0) -> np.ndarray:
    """Create mask for road region: skip top y_frac, keep rest. Default: lower 75%."""
    mask = np.zeros((h, w), dtype=np.uint8)
    y1 = int(h * y_frac)
    y2 = int(h * bottom_frac)
    mask[y1:y2, :] = 255
    return mask


# -----------------------------------------------------------------------------
# Motion blob detection (with ROI, chroma-based shadow suppression, edge filter)
# -----------------------------------------------------------------------------


class MotionBlobDetector:
    """MOG2 + ROI + morphology + stricter filtering. Chroma-based shadow suppression."""

    def __init__(
        self,
        history: int = 500,
        var_threshold: int = 20,
        min_area: int = 800,
        max_area: int = 60_000,
        morph_kernel: tuple[int, int] = (5, 5),
        roi_y_frac: float = 0.25,
        aspect_min: float = 0.3,
        aspect_max: float = 4.0,
        use_chroma: bool = True,
        edge_overlap_thresh: float = 0.12,
    ):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=True,
        )
        self.min_area = min_area
        self.max_area = max_area
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, morph_kernel)
        self.roi_y_frac = roi_y_frac
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.use_chroma = use_chroma
        self.edge_overlap_thresh = edge_overlap_thresh
        self._roi_cache: dict[tuple[int, int], np.ndarray] = {}

    def _get_roi_for_frame(self, h: int, w: int) -> np.ndarray:
        key = (h, w)
        if key not in self._roi_cache:
            self._roi_cache[key] = make_road_roi_mask(h, w, y_frac=self.roi_y_frac, bottom_frac=1.0)
        return self._roi_cache[key]

    def detect(self, frame: np.ndarray) -> list[tuple[float, float, float, float]]:
        h, w = frame.shape[:2]
        roi = self._get_roi_for_frame(h, w)

        # Luminance (grayscale): best for daytime vehicle/road contrast; chroma for night/shadow
        if self.use_chroma:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            _, a, b = cv2.split(lab)
            chroma = cv2.addWeighted(a, 0.5, b, 0.5, 0)
            fg = self.bg_sub.apply(chroma)
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fg = self.bg_sub.apply(gray)

        fg[fg == 127] = 0  # Remove shadow mask
        fg = cv2.bitwise_and(fg, roi)  # Apply ROI

        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.morph_kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, self.morph_kernel)

        # Edge agreement: blobs should overlap with edges (cars have edges; glare often doesn't)
        edges_bin = None
        if self.edge_overlap_thresh > 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edges_in_roi = cv2.bitwise_and(edges, roi)
            edges_bin = (edges_in_roi > 0).astype(np.uint8)

        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg, connectivity=8)
        bboxes = []
        for i in range(1, nlabels):
            area = stats[i, cv2.CC_STAT_AREA]
            if not (self.min_area <= area <= self.max_area):
                continue

            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            bw = stats[i, cv2.CC_STAT_WIDTH]
            bh = stats[i, cv2.CC_STAT_HEIGHT]

            # Aspect ratio: vehicles are typically wider than tall (or similar)
            if bh > 0:
                ar = bw / bh
                if not (self.aspect_min <= ar <= self.aspect_max):
                    continue

            # Edge overlap filter: blob region should have some edge density
            if self.edge_overlap_thresh > 0 and edges_bin is not None:
                blob_mask = (labels == i).astype(np.uint8)
                overlap = cv2.bitwise_and(blob_mask, edges_bin)
                edge_ratio = overlap.sum() / (blob_mask.sum() + 1e-6)
                if edge_ratio < self.edge_overlap_thresh:
                    continue

            bboxes.append((float(x), float(y), float(x + bw), float(y + bh)))
        return bboxes


# -----------------------------------------------------------------------------
# SORT-like tracker
# -----------------------------------------------------------------------------


class BlobTracker:
    """Track motion blobs with Kalman + IoU matching. Reliability gating: prune static/huge+static."""

    def __init__(
        self,
        max_age: int = 12,
        min_hits: int = 5,
        iou_threshold: float = 0.3,
        min_speed_px_s: float = 5.0,
        max_static_age: int = 15,
        min_speed_for_huge: float = 3.0,
        huge_area: float = 15000.0,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.min_speed_px_s = min_speed_px_s
        self.max_static_age = max_static_age  # Prune tracks that stay static this long
        self.min_speed_for_huge = min_speed_for_huge  # Huge + slow = glare
        self.huge_area = huge_area
        self.tracks: list[Track] = []
        self.next_id = 1
        self.dt = 1.0 / 30.0

    def set_fps(self, fps: float) -> None:
        self.dt = 1.0 / fps if fps > 0 else 1.0 / 30.0

    def set_dt(self, dt: float) -> None:
        self.dt = dt

    def _is_reliable(self, t: Track) -> bool:
        """Drop huge+static (glare) and long-static tracks from matching."""
        if t.hits < self.min_hits:
            return True  # Let new tracks try
        if t.area > self.huge_area and t.speed_px_s < self.min_speed_for_huge:
            return False  # Glare/illumination blob
        if t.hits > 10 and t.speed_px_s < self.min_speed_px_s:
            return False  # Long-standing static
        return True

    def update(self, dets: list[tuple[float, float, float, float]], frame_idx: int) -> list[Track]:
        # Prune unreliable tracks
        self.tracks = [t for t in self.tracks if self._is_reliable(t)]
        # Predict
        for t in self.tracks:
            t.kalman.predict(self.dt)
            t.time_since_update += 1

        if not dets:
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.time_since_update == 0 and t.hits >= self.min_hits]

        # Match (only reliable tracks)
        reliable = [t for t in self.tracks if self._is_reliable(t)]
        cost = iou_cost_matrix(reliable, dets)
        r, c = linear_sum_assignment(cost)
        matched = [(r[i], c[i]) for i in range(len(r)) if cost[r[i], c[i]] < (1 - self.iou_threshold)]

        matched_track_idx = set()
        matched_det_idx = set()
        for ti, di in matched:
            matched_track_idx.add(ti)
            matched_det_idx.add(di)

        # Update matched (ti indexes into reliable)
        for ti, di in matched:
            t = reliable[ti]
            z = np.array([
                (dets[di][0] + dets[di][2]) / 2,
                (dets[di][1] + dets[di][3]) / 2,
                dets[di][2] - dets[di][0],
                dets[di][3] - dets[di][1],
            ])
            t.kalman.update(z)
            t.bbox = t.kalman.to_bbox()
            t.hits += 1
            t.time_since_update = 0
            vx, vy = t.kalman.velocity
            t.update_kinematics(self.dt, vx, vy)

        # Unmatched tracks (predict-only; still update kinematics from predicted vel)
        for ti in range(len(reliable)):
            if ti not in matched_track_idx:
                t = reliable[ti]
                t.bbox = t.kalman.to_bbox()
                vx, vy = t.kalman.velocity
                t.update_kinematics(self.dt, vx, vy)

        # Create new tracks for unmatched detections
        for di in range(len(dets)):
            if di not in matched_det_idx:
                x1, y1, x2, y2 = dets[di]
                k = KalmanBox.from_bbox(x1, y1, x2, y2)
                t = Track(track_id=self.next_id, bbox=(x1, y1, x2, y2), kalman=k)
                self.next_id += 1
                self.tracks.append(t)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return active tracks: recently updated, enough hits. Velocity filter: exclude
        # tracks that have been around (hits>5) but are near-static (likely buildings/glare).
        active = []
        for t in self.tracks:
            if t.time_since_update != 0 or t.hits < self.min_hits:
                continue
            if t.hits > 5 and t.speed_px_s < self.min_speed_px_s:
                continue  # Static blob - skip
            active.append(t)
        return active


# -----------------------------------------------------------------------------
# Incident classification
# -----------------------------------------------------------------------------


@dataclass
class CandidateScore:
    """Event candidate score and evidence. Stage 1 output; 'send_to_vlm' = hard threshold."""

    event_candidate: bool
    send_to_vlm: bool  # Binary: score >= threshold
    score: float
    threshold: float
    event_types: dict  # e.g. stopped_vehicle, shockwave, pedestrian, wrong_way, debris_like
    contributions: dict


# COCO vehicle classes: car=2, motorcycle=3, bus=5, truck=7
YOLO_VEHICLE_CLASSES = [2, 3, 5, 7]
YOLO_PERSON_CLASS = 0

# Default weights (tune per deployment)
DEFAULT_WEIGHTS = {
    "decel_spike": 1.5,
    "stopped_in_lane": 1.5,
    "shockwave": 1.5,
    "yolo_person_detected": 2.5,
    "high_track_count": 1.0,
}
DEFAULT_HIGH_TRACK_THRESH = 20  # tracks > this contributes
DEFAULT_THRESHOLD = 4.0
DEFAULT_TARGET_FPS = 5.0
DEFAULT_YOLO_HZ = 1.0  # Run YOLO at 1 Hz when sparse
PERSON_CONF_THRESH = 0.5  # Require higher conf for person to reduce false positives
PERSON_PERSIST_RUNS = 2   # Person must appear in N consecutive YOLO runs to trust
STOPPED_THRESH_PX_S = 5.0
STOPPED_PERSIST_FRAMES = 8  # ~1.6s at 5 FPS
DECEL_MIN_HITS = 5  # Only use decel/accel from tracks present 5+ frames
SHOCKWAVE_MED_DROP_THRESH = 8.0  # px/s (median_speed_drop)
SHOCKWAVE_DENSITY_INCREASE = 0.3  # fraction


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


class MotionFirstTracker:
    """
    Motion-first incident detection.

    - Motion blobs: primary tracking
    - YOLO: every frame, person + vehicle detection
    - Processing at target_fps (default 5)
    """

    def __init__(
        self,
        video_source: str | int,
        yolo_model: str = "yolo26s.pt",
        incident_threshold: float = DEFAULT_THRESHOLD,
        weights: dict | None = None,
        output_video: str | None = None,
        device: str = "mps",
        roi_y_frac: float = 0.25,
        target_fps: float = DEFAULT_TARGET_FPS,
    ):
        self.video_source = video_source
        self.incident_threshold = incident_threshold
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.device = device
        self.roi_y_frac = roi_y_frac
        self.target_fps = target_fps

        # Video
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_source}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Motion: use luminance (grayscale) for daytime—chroma misses vehicle/road contrast
        self.blob_detector = MotionBlobDetector(
            min_area=150,
            max_area=50_000,
            var_threshold=10,
            morph_kernel=(2, 2),
            roi_y_frac=roi_y_frac,
            aspect_min=0.2,
            aspect_max=5.0,
            use_chroma=False,
            edge_overlap_thresh=0.0,
        )
        self.tracker = BlobTracker(
            max_age=18, min_hits=2, iou_threshold=0.2, min_speed_px_s=2.0,
            max_static_age=20, min_speed_for_huge=2.0, huge_area=35_000,
        )
        self.tracker.set_fps(self.target_fps)

        # YOLO (lazy)
        self._yolo = None
        self._yolo_model_path = yolo_model

        # State: shockwave
        self._median_speed_history: deque = deque(maxlen=30)
        self._track_count_history: deque = deque(maxlen=30)
        # State: stopped-in-lane (per-track stop frames)
        self._stop_frames: dict[int, int] = {}
        self._decel_history: deque = deque(maxlen=10)
        self._grid_size = 20
        self._last_person_result = False
        self._last_yolo_vehicle_count = 0
        self._person_run_count = 0  # Consecutive YOLO runs with person detected
        self._last_candidate: CandidateScore | None = None

        # Output
        self.output_video_path = output_video
        self.video_writer = None
        if output_video:
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_writer = cv2.VideoWriter(
                output_video, cv2.VideoWriter_fourcc(*"mp4v"), self.target_fps, (w, h)
            )

    @property
    def yolo(self):
        if self._yolo is None:
            from ultralytics import YOLO
            self._yolo = YOLO(self._yolo_model_path)
        return self._yolo

    def _pixel_to_grid(self, cx: float, cy: float) -> tuple[int, int]:
        return int(cx // self._grid_size), int(cy // self._grid_size)

    def _indicator_decel_spike(self, active_tracks: list, weight: float) -> tuple[float, str]:
        """Hard decel: max negative accel < -15 px/s². Only tracks with hits >= DECEL_MIN_HITS."""
        if not active_tracks:
            return 0.0, ""
        count = 0
        for t in active_tracks:
            if t.hits < DECEL_MIN_HITS:
                continue
            if len(t.accel_history) >= 2:
                accels = list(t.accel_history)
                min_accel = min(accels[-5:]) if len(accels) >= 5 else min(accels)
                if min_accel < -15:
                    count += 1
        raw = min(count * 0.5, 2.0)
        contrib = weight * raw
        event = "decel" if raw > 0 else ""
        return contrib, event

    def _indicator_stopped_in_lane(self, active_tracks: list, weight: float) -> tuple[float, str]:
        """Speed < thresh for > T seconds. Lane zone = ROI (no shoulder)."""
        STOPPED = STOPPED_THRESH_PX_S
        PERSIST = STOPPED_PERSIST_FRAMES
        count = 0
        for t in active_tracks:
            if t.speed_px_s < STOPPED:
                self._stop_frames[t.track_id] = self._stop_frames.get(t.track_id, 0) + 1
            else:
                self._stop_frames[t.track_id] = 0
            if self._stop_frames.get(t.track_id, 0) >= PERSIST:
                count += 1
        contrib = weight * min(count, 2) / 2.0
        event = "stopped_vehicle" if count > 0 else ""
        return contrib, event

    def _indicator_shockwave(
        self, active_tracks: list, weight: float
    ) -> tuple[float, str]:
        """median_speed_drop large AND density_increase AND persists."""
        if len(active_tracks) == 0:
            self._median_speed_history.append(0.0)
            self._track_count_history.append(0)
            return 0.0, ""
        med_speed = float(np.median([t.speed_px_s for t in active_tracks]))
        track_count = len(active_tracks)
        self._median_speed_history.append(med_speed)
        self._track_count_history.append(track_count)
        if len(self._median_speed_history) < 5:
            return 0.0, ""
        prev_med = self._median_speed_history[-5]
        curr_med = self._median_speed_history[-1]
        prev_count = self._track_count_history[-5]
        curr_count = self._track_count_history[-1]
        med_drop = max(0, prev_med - curr_med)
        density_inc = (curr_count - prev_count) / max(prev_count, 1) if prev_count > 0 else 0
        if med_drop >= SHOCKWAVE_MED_DROP_THRESH and density_inc >= SHOCKWAVE_DENSITY_INCREASE:
            contrib = weight
            event = "shockwave"
        else:
            contrib = 0.0
            event = ""
        return contrib, event

    def _run_yolo(self, frame: np.ndarray) -> tuple[bool, int]:
        """Run YOLO predict (not track) on frame. Sparse: call at 1 Hz or on pre-threshold."""
        person_found = False
        vehicle_count = 0
        try:
            res = self.yolo.predict(
                frame,
                conf=0.25,
                classes=[YOLO_PERSON_CLASS] + YOLO_VEHICLE_CLASSES,
                max_det=50,
                device=self.device,
                verbose=False,
            )
            if res and len(res) > 0 and res[0].boxes is not None and res[0].boxes.cls is not None:
                cls = res[0].boxes.cls.cpu().numpy()
                conf = res[0].boxes.conf.cpu().numpy() if res[0].boxes.conf is not None else np.ones_like(cls)
                person_mask = (cls == YOLO_PERSON_CLASS) & (conf >= PERSON_CONF_THRESH)
                person_found = bool(np.any(person_mask))
                vehicle_count = int(np.sum(np.isin(cls, YOLO_VEHICLE_CLASSES)))
            # Require persistence: person must appear in N consecutive YOLO runs
            if person_found:
                self._person_run_count = min(self._person_run_count + 1, PERSON_PERSIST_RUNS)
            else:
                self._person_run_count = 0
            self._last_person_result = self._person_run_count >= PERSON_PERSIST_RUNS
            self._last_yolo_vehicle_count = vehicle_count
        except Exception:
            self._person_run_count = 0
            self._last_person_result = False
            self._last_yolo_vehicle_count = 0
        return self._last_person_result, self._last_yolo_vehicle_count

    def _classify_candidate(
        self,
        frame: np.ndarray,
        frame_idx: int,
        active_tracks: list,
    ) -> CandidateScore:
        """Stage 1: event candidate scoring. send_to_vlm = hard threshold for VLM trigger."""
        contributions = {}
        event_types = {}

        # Clean stale stop_frames
        active_ids = {t.track_id for t in active_tracks}
        self._stop_frames = {k: v for k, v in self._stop_frames.items() if k in active_ids}

        # A) Decel spike (averaged over last 10 frames)
        c_decel_raw, _ = self._indicator_decel_spike(
            active_tracks, self.weights.get("decel_spike", 0.0)
        )
        self._decel_history.append(c_decel_raw)
        c_decel = sum(self._decel_history) / len(self._decel_history) if self._decel_history else 0.0
        contributions["decel_spike"] = c_decel
        if c_decel > 0:
            event_types["decel"] = c_decel

        # B) Stopped-in-lane
        c_stopped, e_stopped = self._indicator_stopped_in_lane(
            active_tracks, self.weights.get("stopped_in_lane", 0.0)
        )
        contributions["stopped_in_lane"] = c_stopped
        if e_stopped:
            event_types["stopped_vehicle"] = c_stopped

        # C) Shockwave
        c_shock, e_shock = self._indicator_shockwave(
            active_tracks, self.weights.get("shockwave", 0.0)
        )
        contributions["shockwave"] = c_shock
        if e_shock:
            event_types["shockwave"] = c_shock

        # D) YOLO person
        person = self._last_person_result
        c_person = self.weights.get("yolo_person_detected", 0.0) if person else 0.0
        contributions["yolo_person_detected"] = c_person
        if person:
            event_types["pedestrian"] = c_person

        # E) High track count: tracks > thresh contributes
        track_n = len(active_tracks)
        c_high = self.weights.get("high_track_count", 0.0) if track_n > DEFAULT_HIGH_TRACK_THRESH else 0.0
        contributions["high_track_count"] = c_high

        score = sum(contributions.values())
        send_to_vlm = score >= self.incident_threshold
        event_candidate = len(event_types) > 0
        result = CandidateScore(
            event_candidate=event_candidate,
            send_to_vlm=send_to_vlm,
            score=score,
            threshold=self.incident_threshold,
            event_types=event_types,
            contributions=contributions,
        )
        self._last_candidate = result
        return result

    def run(self, display: bool = True, check_interval: int = 10):
        """Run at target_fps with frame skipping. YOLO sparse (1 Hz)."""
        skip = max(1, int(round(self.fps / self.target_fps)))
        dt = skip / self.fps
        self.tracker.set_dt(dt)
        frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.2
        yolo_interval = max(1, int(self.target_fps / DEFAULT_YOLO_HZ)) if DEFAULT_YOLO_HZ > 0 else 1
        print(f"[motion_first] input_fps={self.fps:.1f} target={self.target_fps} skip={skip} YOLO every {yolo_interval} processed")
        frame_idx = 0
        processed_idx = 0
        last_process_time = time.perf_counter()
        fps_times: deque = deque(maxlen=30)

        while self.cap.isOpened():
            ok, frame = self.cap.read()
            if not ok:
                break
            if frame_idx % skip != 0:
                frame_idx += 1
                continue

            # 1. YOLO sparse (1 Hz or yolo_interval processed frames)
            if processed_idx % yolo_interval == 0:
                self._run_yolo(frame)

            # 2. Motion blobs
            dets = self.blob_detector.detect(frame)
            active_tracks = self.tracker.update(dets, processed_idx)

            # 4. Candidate scoring (every frame)
            cand = self._classify_candidate(
                frame, processed_idx, active_tracks=active_tracks
            )
            parts = [f"{k}:{v:.2f}" for k, v in cand.contributions.items()]
            print(f"[motion_first] frame {frame_idx} | score={cand.score:.2f} | {', '.join(parts)}")
            if cand.send_to_vlm:
                print(f"  >> CANDIDATE send_to_vlm | events: {cand.event_types}")

            # 5. Draw
            if display or self.video_writer:
                vis = frame.copy()
                # Draw ROI boundary line (road region starts below)
                h, w = vis.shape[:2]
                y_roi = int(h * self.roi_y_frac)
                cv2.line(vis, (0, y_roi), (w, y_roi), (100, 200, 255), 2)
                cv2.putText(vis, "ROI", (10, y_roi - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
                for t in active_tracks:
                    x1, y1, x2, y2 = map(int, t.bbox)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        vis, f"#{t.track_id} {t.speed_px_s:.0f}px/s",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                    )
                if self._last_candidate:
                    cand = self._last_candidate
                    color = (0, 0, 255) if cand.send_to_vlm else (0, 255, 0)
                    label = "VLM" if cand.send_to_vlm else "OK"
                    cv2.putText(vis, f"{label} ({cand.score:.1f})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    short = {"decel_spike": "decel", "stopped_in_lane": "stop", "shockwave": "shock", "yolo_person_detected": "pers", "high_track_count": "tracks"}
                    parts = [f"{short.get(k, k[:4])}:{v:.1f}" for k, v in cand.contributions.items() if v > 0]
                    if parts:
                        cv2.putText(vis, " | ".join(parts), (10, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)
                motion_n = len(active_tracks)
                cv2.putText(vis, f"YOLO: {self._last_yolo_vehicle_count} | tracks: {motion_n}",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                # FPS (rolling average)
                fps_times.append(time.perf_counter())
                if len(fps_times) >= 2:
                    fps = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])
                    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                if self.video_writer:
                    self.video_writer.write(vis)
                if display:
                    cv2.imshow("MotionFirst", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1
            processed_idx += 1

            # Throttle to target FPS
            elapsed = time.perf_counter() - last_process_time
            sleep_time = frame_interval - elapsed
            if sleep_time > 0.001:
                time.sleep(sleep_time)
            last_process_time = time.perf_counter()

        self.cap.release()
        if self.video_writer:
            self.video_writer.release()
        if display:
            cv2.destroyAllWindows()
        print(f"[motion_first] Done. Read {frame_idx} frames, processed {processed_idx}.")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main():
    import argparse
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _default_video = os.path.join(_script_dir, "test-video-processing", "2026-01-31 23-11-57.mov")
    p = argparse.ArgumentParser(description="Motion-first incident detection")
    p.add_argument("source", nargs="?", default=_default_video, help="Video file or stream URL")
    p.add_argument("--yolo", default="yolo26s.pt", help="YOLO model path")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Incident score threshold")
    p.add_argument("--no-display", action="store_true", help="Disable display window")
    p.add_argument("--output", "-o", help="Output video path")
    p.add_argument("--device", default="mps", help="YOLO device (mps/cuda/cpu)")
    p.add_argument("--roi-y-frac", type=float, default=0.25,
                   help="Skip top fraction of frame for ROI (0.25 = road in lower 75%%)")
    p.add_argument("--target-fps", type=float, default=DEFAULT_TARGET_FPS,
                   help="Process at this FPS (default 5)")
    args = p.parse_args()

    tracker = MotionFirstTracker(
        video_source=args.source,
        yolo_model=args.yolo,
        incident_threshold=args.threshold,
        output_video=args.output,
        device=args.device,
        roi_y_frac=args.roi_y_frac,
        target_fps=args.target_fps,
    )
    tracker.run(display=not args.no_display)


if __name__ == "__main__":
    main()
