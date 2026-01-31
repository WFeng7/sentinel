# TODO: implement writing tool that captures a sliding window of the video

import cv2
import numpy as np
from ultralytics import YOLO

from ultralytics.utils.plotting import Annotator, colors

class ObjectTracker: 
    def __init__(self, model: str, url: str, conf_threshold: float = 0.01, iou_threshold: float = 0.9, imgsz: int = 1280):
        self.model = YOLO(model) # either modelname or path to model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        # opening stream
        self.cap = cv2.VideoCapture(url)
        assert self.cap.isOpened(), "Could not open stream"

        # padding: extra pixels around bbox when cropping (crop is larger than bbox)
        self.crop_padding = self.image_padding = 8

        self.crop_size = (100, 100)

        # display settings
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        assert self.fps and self.fps > 0, "FPS is 0" 
        self.delay = int(1000 / self.fps)

        # window setup
        self.window_name = "Object Tracking"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        self.current_data = None
        self.selected_id = None
        self.ann = None
        self.current_frame_shape = None  # (h, w) for click coordinate conversion

        # Minimum padding (pixels) for hit area so small boxes are easier to click
        self.click_padding = 8

        # Preprocessing for snow/glare (Providence cameras): CLAHE + mild highlight compression
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._preprocess_gamma = 1.00  # mild gamma on luminance to compress snow glare; >1 darkens brights

        # Velocity: track history (id -> list of (frame_idx, cx, cy)); max length per track
        self._track_history: dict[int, list[tuple[int, float, float]]] = {}
        self._track_history_max_len = 15
        # Speed -> color: R = slow, Y = medium, G = fast (BGR)
        self._speed_slow_thresh = 15.0   # px/s below this -> Red
        self._speed_fast_thresh = 45.0   # px/s above this -> Green; between -> Yellow

    def _speed_to_color_bgr(self, speed_px_per_sec: float) -> tuple[int, int, int]:
        """Map speed (px/s) to BGR: Red = slowest, Yellow = medium, Green = fastest."""
        if speed_px_per_sec <= self._speed_slow_thresh:
            return (0, 0, 255)   # Red
        if speed_px_per_sec >= self._speed_fast_thresh:
            return (0, 255, 0)   # Green
        # Yellow in between (linear blend R -> Y -> G would need two segments; keep simple: mid = Yellow)
        return (0, 255, 255)   # Yellow

    def _update_velocity(self, track_id: int, cx: float, cy: float, frame_idx: int) -> float:
        """Append (frame_idx, cx, cy) to history; return speed in px/s (0 if not enough data)."""
        tid = int(track_id)
        if tid not in self._track_history:
            self._track_history[tid] = []
        hist = self._track_history[tid]
        hist.append((frame_idx, cx, cy))
        if len(hist) > self._track_history_max_len:
            hist.pop(0)
        if len(hist) < 2 or self.fps <= 0:
            return 0.0
        (f0, x0, y0), (f1, x1, y1) = hist[-2], hist[-1]
        dt_frames = f1 - f0
        if dt_frames <= 0:
            return 0.0
        dt_sec = dt_frames / self.fps
        dist = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        return float(dist / dt_sec)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Lightweight glare/contrast normalization before YOLO: CLAHE on L + mild gamma on L.
        Helps with snow scenes and overexposure; run this frame through YOLO, draw on original for display.
        """
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        # Mild gamma on luminance to compress highlights (snow) without overdoing it
        l_float = np.clip(l.astype(np.float32) / 255.0, 0, 1)
        l_gamma = np.power(l_float, self._preprocess_gamma)
        l = (l_gamma * 255.0).astype(np.uint8)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _window_to_image_coords(self, x, y):
        """Convert window click (x,y) to image coordinates (handles scaled/resized window)."""
        if self.current_frame_shape is None:
            return x, y
        try:
            rect = cv2.getWindowImageRect(self.window_name)
            if rect is None or rect[2] <= 0 or rect[3] <= 0:
                return x, y
            win_w, win_h = rect[2], rect[3]
        except Exception:
            return x, y
        frame_h, frame_w = self.current_frame_shape
        scale = min(win_w / frame_w, win_h / frame_h)
        disp_w, disp_h = frame_w * scale, frame_h * scale
        x_off = (win_w - disp_w) / 2
        y_off = (win_h - disp_h) / 2
        x_im = (x - x_off) / scale
        y_im = (y - y_off) / scale
        return x_im, y_im

    def mouse_callback(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if not hasattr(self, 'current_data') or self.current_data is None:
            return
        x_im, y_im = self._window_to_image_coords(x, y)
        boxes, ids = self.current_data
        self.selected_id = None
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # Expand hit area by padding so small boxes are clickable
            pad = self.click_padding
            if (x1 - pad <= x_im <= x2 + pad) and (y1 - pad <= y_im <= y2 + pad):
                self.selected_id = int(ids[i].item())
                break
    
    def crop_object(self, im, box):
        h, w = im.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        pad = self.crop_padding
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        cropped = im[y1:y2, x1:x2]

        ch, cw = cropped.shape[:2]
        scale = min(self.crop_size[0] / ch, self.crop_size[1] / cw)
        return cv2.resize(cropped, (int(cw * scale), int(ch * scale)))
    
    def add_crop_overlap(self, im, crop):
        if crop is None: 
            return im
        h, w = im.shape[:2]
        y1, x1 = min(h, self.image_padding), max(0, w - crop.shape[1] - self.image_padding)
        y2, x2 = y1 + crop.shape[0], x1 + crop.shape[1]

        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        im[y1:y2, x1:x2] = crop

        return im
    
    def process_selected_object(self, im, boxes, ids):
        if self.selected_id is None:
            return im
        for i, id in enumerate(ids):
            tid = int(id.item() if hasattr(id, 'item') else id)
            if tid == self.selected_id:
                crop = self.crop_object(im.copy(), boxes[i])
                return self.add_crop_overlap(im, crop)
        
        print(f"Object {self.selected_id} lost from tracking")
        self.selected_id = None
        return im

    def run(self):
        print("Click on any object to track")
        print("Press 'q' to quit")
        print("Press 'c' to clear selection")

        frame_idx = 0
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                print("Error reading frame")
                break
            self.current_frame_shape = frame.shape[:2]  # for click -> image coordinate conversion
            # Run YOLO on preprocessed frame (CLAHE + gamma for snow/glare); draw on original
            frame_for_yolo = self.preprocess_frame(frame)
            # Vehicle classes only (COCO: car, motorcycle, bus, truck) for car-like / truck-like / motorcycle-like
            results = self.model.track(
                frame_for_yolo,
                persist=True,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                classes=[2, 3, 5, 7],
                max_det=500,
                tracker="botsort.yaml"
            )
            self.ann = Annotator(frame, line_width=2, font_size=1)

            n_det = 0
            mean_conf = 0.0
            min_box_area = 0.0
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None and result.boxes.id is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    ids = result.boxes.id.cpu()
                    self.current_data = (boxes, ids)
                    frame = self.process_selected_object(frame, boxes, ids)

                    n_det = len(boxes)
                    if result.boxes.conf is not None and n_det > 0:
                        mean_conf = float(result.boxes.conf.cpu().numpy().mean())
                    if n_det > 0:
                        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                        min_box_area = float(np.min(areas))

                    if boxes is not None or ids is not None:
                        for box, id in zip(boxes, ids):
                            cx = (box[0] + box[2]) / 2.0
                            cy = (box[1] + box[3]) / 2.0
                            speed_px_s = self._update_velocity(int(id), cx, cy, frame_idx)
                            color_bgr = self._speed_to_color_bgr(speed_px_s)
                            self.ann.box_label(box, f"{speed_px_s:.0f} px/s", color=color_bgr)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_luminance = float(np.mean(gray))
            print(f"frame {frame_idx}: n_det={n_det} mean_conf={mean_conf:.3f} min_box_area={min_box_area:.0f} mean_luminance={mean_luminance:.1f}")
            frame_idx += 1

            cv2.imshow("Object Tracking", frame)
            key = cv2.waitKey(self.delay) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self.selected_id = None
                print("Selection cleared")

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use yolo11m.pt or yolo11l.pt for more detections on distant/small cars (slower).
    tracker = ObjectTracker(
        model="yolo26s.pt",
        url="https://cdn3.wowza.com/1/ZTdLZmtEVnB1aVEz/M3lZck51/hls/live/playlist.m3u8",
    )
    tracker.run()