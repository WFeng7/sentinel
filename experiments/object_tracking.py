# TODO: implement writing tool that captures a sliding window of the video

import cv2
import json
import math
import os
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from transformers import pipeline
from PIL import Image

class ObjectTracker: 
    def __init__(self, model: str, video_source: str, conf_threshold: float = 0.01, iou_threshold: float = 0.9, imgsz: int = 1280, database_file: str = "speed_database.json", grid_size: int = 20, enable_crash_detection: bool = True, crash_check_interval: int = 30, crash_indicator_threshold: float = 0.75, output_video: str = None):
        """
        Initialize ObjectTracker.
        
        Args:
            model: Path to YOLO model file (e.g., "yolo26s.pt")
            video_source: Video source - can be:
                - URL string (e.g., "https://...")
                - File path (e.g., "/path/to/video.mp4", "video.mp4")
                - Integer for webcam (e.g., 0, 1)
            conf_threshold: Detection confidence threshold
            iou_threshold: IoU threshold for NMS
            imgsz: Image size for YOLO inference
            database_file: Database file name (for compatibility, not used)
            grid_size: Grid cell size in pixels
            enable_crash_detection: Enable crash detection model
            crash_check_interval: Check for crashes every N frames
            crash_indicator_threshold: Number of indicators that must be positive to declare a crash (default: 2)
            output_video: Optional path to save output video (e.g., "output.mp4")
        """
        self.model = YOLO(model) # either modelname or path to model
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz

        # Crash detection setup
        self.enable_crash_detection = enable_crash_detection
        self.crash_check_interval = crash_check_interval  # Check every N frames
        self.crash_indicator_threshold = crash_indicator_threshold  # Confidence to declare a crash
        
        # Initialize crash detection indicators
        self.crash_pipe = None
        self.enable_huggingface_indicator = True  # Toggle individual indicators
        if self.enable_crash_detection:
            if self.enable_huggingface_indicator:
                print("Loading crash detection model (Hugging Face)...")
                try:
                    self.crash_pipe = pipeline("image-classification", model="tiya1012/vit-accident-image")
                    print("Crash detection model loaded successfully")
                except Exception as e:
                    print(f"Warning: Could not load crash detection model: {e}")
                    print("Hugging Face indicator will be disabled")
                    self.enable_huggingface_indicator = False
                    self.crash_pipe = None
        
        # Crash detection state
        self.last_crash_result = None
        self.last_crash_indicators = {}  # Store last indicator results for debugging

        # Determine if video_source is a file, URL, or webcam
        self.video_source = video_source
        self.is_file = self._is_video_file(video_source)
        self.is_url = self._is_url(video_source)
        self.is_webcam = isinstance(video_source, int) or (isinstance(video_source, str) and video_source.isdigit())
        
        # Opening video source
        if self.is_webcam:
            self.cap = cv2.VideoCapture(int(video_source))
        else:
            self.cap = cv2.VideoCapture(video_source)
        
        if not self.cap.isOpened():
            if self.is_file:
                raise ValueError(f"Could not open video file: {video_source}")
            elif self.is_url:
                raise ValueError(f"Could not open video stream URL: {video_source}")
            else:
                raise ValueError(f"Could not open video source: {video_source}")
        
        # Print source information
        if self.is_file:
            print(f"Video file loaded: {video_source}")
        elif self.is_url:
            print(f"Video stream URL: {video_source}")
        else:
            print(f"Video source: {video_source}")
        
        # Video output setup
        self.output_video_path = output_video
        self.video_writer = None
        if self.output_video_path:
            self._setup_video_writer()

        # padding: extra pixels around bbox when cropping (crop is larger than bbox)
        self.crop_padding = self.image_padding = 8

        self.crop_size = (100, 100)

        # display settings
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        # For video files, FPS might be 0 or invalid, use default
        if self.fps <= 0 or not self.fps:
            if self.is_file:
                print(f"Warning: Could not determine FPS from video file, using default 30 FPS")
                self.fps = 30.0
            else:
                raise ValueError("FPS is 0 or invalid")
        
        # Calculate delay to match video FPS for proper playback speed
        # For video files, use minimal delay (1ms) to allow processing to control speed
        # Processing time will naturally control playback speed
        # For live streams, match FPS to maintain real-time playback
        if self.is_file:
            self.delay = 1  # Minimal delay for video files - processing time controls playback speed
            print(f"Video file detected: FPS={self.fps:.2f}, using minimal delay (1ms) for fast playback")
        else:
            self.delay = max(1, int(1000 / self.fps))  # Match FPS for live streams
            print(f"Live stream: FPS={self.fps:.2f}, delay={self.delay}ms")

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
        
        # Speed database: maps grid cell (grid_x, grid_y) to list of speeds recorded in that cell
        # Key: (grid_x, grid_y) - grid cell coordinates (each cell is grid_size x grid_size pixels)
        # Value: list of speeds (px/s) recorded in that grid cell (keeps last 50 instances per cell)
        self._database_file = database_file
        self._grid_size = grid_size
        self._speed_database: dict[tuple[int, int], list[float]] = {}
        self._speed_database_max_len = 50  # Keep only last 50 speed measurements per grid cell
        # Acceleration database: maps grid cell (grid_x, grid_y) to list of accelerations recorded in that cell
        # Value: list of accelerations (px/s²) recorded in that grid cell (keeps last 50 instances per cell)
        self._acceleration_database: dict[tuple[int, int], list[float]] = {}
        self._acceleration_database_max_len = 50  # Keep only last 50 acceleration measurements per grid cell
        # Jerk database: maps grid cell (grid_x, grid_y) to list of jerks recorded in that cell
        # Value: list of jerks (px/s³) recorded in that grid cell (keeps last 50 instances per cell)
        self._jerk_database: dict[tuple[int, int], list[float]] = {}
        self._jerk_database_max_len = 50  # Keep only last 50 jerk measurements per grid cell
        self._show_grid = False  # Toggle grid visualization
        
        # Speed history per track ID for acceleration calculation: track_id -> list of (frame_idx, speed)
        self._speed_history: dict[int, list[tuple[int, float]]] = {}
        self._speed_history_max_len = 10  # Keep last 10 speed measurements per track
        # Acceleration history per track ID for jerk calculation: track_id -> list of (frame_idx, acceleration)
        self._acceleration_history: dict[int, list[tuple[int, float]]] = {}
        self._acceleration_history_max_len = 10  # Keep last 10 acceleration measurements per track
        
        # Both speeds and accelerations are now in-memory only (no file persistence)
        # Database file parameter kept for backward compatibility but not used

    def _is_video_file(self, source: str) -> bool:
        """Check if source is a video file path."""
        if isinstance(source, int):
            return False
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
        return any(source.lower().endswith(ext) for ext in video_extensions) or os.path.isfile(source)
    
    def _is_url(self, source: str) -> bool:
        """Check if source is a URL."""
        if isinstance(source, int):
            return False
        return source.startswith(('http://', 'https://', 'rtsp://', 'rtmp://'))
    
    def _setup_video_writer(self):
        """Setup video writer for output video file."""
        if not self.output_video_path:
            return
        
        # Get video properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.fps
        
        # Use MP4V codec (works well cross-platform)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            print(f"Warning: Could not initialize video writer for {self.output_video_path}")
            self.video_writer = None
        else:
            print(f"Output video will be saved to: {self.output_video_path}")

    def _pixel_to_grid_cell(self, cx: float, cy: float) -> tuple[int, int]:
        """Convert pixel coordinates to grid cell coordinates.
        
        Args:
            cx: Center x coordinate in pixels
            cy: Center y coordinate in pixels
            
        Returns:
            (grid_x, grid_y): Grid cell coordinates
        """
        grid_x = int(cx // self._grid_size)
        grid_y = int(cy // self._grid_size)
        return (grid_x, grid_y)

    def _draw_grid(self, frame: np.ndarray):
        """Draw grid overlay on the frame to visualize grid cells.
        
        Args:
            frame: Frame to draw grid on (modified in place)
        """
        if not self._show_grid:
            return
        
        h, w = frame.shape[:2]
        color = (128, 128, 128)  # Gray color for grid lines
        thickness = 1
        
        # Draw vertical lines
        for x in range(0, w, self._grid_size):
            cv2.line(frame, (x, 0), (x, h), color, thickness)
        
        # Draw horizontal lines
        for y in range(0, h, self._grid_size):
            cv2.line(frame, (0, y), (w, y), color, thickness)
        
        # Highlight grid cells that have speed data (cyan)
        if self._speed_database:
            speed_color = (0, 255, 255)  # Cyan for speed data
            thickness = 2
            for (grid_x, grid_y), speeds in self._speed_database.items():
                if speeds:  # Only highlight cells with data
                    x1 = grid_x * self._grid_size
                    y1 = grid_y * self._grid_size
                    x2 = x1 + self._grid_size
                    y2 = y1 + self._grid_size
                    # Draw rectangle outline for cells with speed data
                    cv2.rectangle(frame, (x1, y1), (x2, y2), speed_color, thickness)
        
        # Highlight grid cells that have acceleration data (magenta)
        if self._acceleration_database:
            accel_color = (255, 0, 255)  # Magenta for acceleration data
            thickness = 2
            for (grid_x, grid_y), accelerations in self._acceleration_database.items():
                if accelerations:  # Only highlight cells with data
                    x1 = grid_x * self._grid_size
                    y1 = grid_y * self._grid_size
                    x2 = x1 + self._grid_size
                    y2 = y1 + self._grid_size
                    # Draw rectangle outline for cells with acceleration data
                    # Offset slightly to show both if cell has both
                    offset = 1
                    cv2.rectangle(frame, (x1 + offset, y1 + offset), (x2 - offset, y2 - offset), accel_color, thickness)

    def _load_database(self):
        """Speeds, accelerations, and jerks are now in-memory only (no file persistence)."""
        # No-op: all databases start empty and are populated during runtime
        print("Speed, acceleration, and jerk databases are in-memory only (no file loading).")

    def _save_database(self):
        """Speeds, accelerations, and jerks are now in-memory only (no file persistence)."""
        total_speed_cells = len(self._speed_database)
        total_speeds = sum(len(speeds) for speeds in self._speed_database.values())
        total_accel_cells = len(self._acceleration_database)
        total_accelerations = sum(len(accels) for accels in self._acceleration_database.values())
        total_jerk_cells = len(self._jerk_database)
        total_jerks = sum(len(jerks) for jerks in self._jerk_database.values())
        """
        print(f"\nDatabase summary (in-memory only, not saved to file):")
        print(f"  Speed grid cells: {total_speed_cells} (max 50 per cell)")
        print(f"  Speed measurements: {total_speeds}")
        print(f"  Acceleration grid cells: {total_accel_cells} (max 50 per cell)")
        print(f"  Acceleration measurements: {total_accelerations}")
        print(f"  Jerk grid cells: {total_jerk_cells} (max 50 per cell)")
        print(f"  Jerk measurements: {total_jerks}")"""

    def _speed_to_color_bgr(self, speed_px_per_sec: float, grid_cell: tuple[int, int]) -> tuple[int, int, int]:
        """Map speed (px/s) to BGR based on z-score from Gaussian MLE of grid cell's historical data.
        
        Args:
            speed_px_per_sec: Current speed in pixels per second
            grid_cell: Grid cell coordinates (grid_x, grid_y)
            
        Returns:
            BGR color tuple:
            - Black if < 3 datapoints in grid cell
            - Green if z-score < -1 (slower than average)
            - Yellow if -1 <= z-score <= 1 (near average)
            - Red if z-score > 1 (faster than average)
        """
        # Check if grid cell has enough data (minimum 3 for reasonable Gaussian estimate)
        if grid_cell not in self._speed_database or len(self._speed_database[grid_cell]) < 3:
            return (0, 0, 0)  # Black - insufficient data
        
        # Get historical speeds for this grid cell
        historical_speeds = self._speed_database[grid_cell]
        
        # Maximum likelihood estimation of Gaussian parameters
        # Mean (μ) = sample mean
        mean = np.mean(historical_speeds)
        
        # Variance (σ²) using MLE: 1/n * sum((x - μ)²)
        variance = np.mean((historical_speeds - mean) ** 2)
        
        # Standard deviation (σ)
        std_dev = np.sqrt(variance)
        
        # Handle edge case where std_dev is 0 (all speeds are identical)
        if std_dev == 0:
            # If all speeds are the same, compare directly
            if speed_px_per_sec < mean:
                return (0, 255, 0)   # Green - slower
            elif speed_px_per_sec > mean:
                return (0, 0, 255)   # Red - faster
            else:
                return (0, 255, 255) # Yellow - same
        
        # Calculate z-score: (x - μ) / σ
        z_score = (speed_px_per_sec - mean) / std_dev
        
        # Color based on z-score thresholds
        if z_score < -1:
            return (0, 255, 0)   # Green - z-score < -1 (slower than average)
        elif z_score > 1:
            return (0, 0, 255)   # Red - z-score > 1 (faster than average)
        else:
            return (0, 255, 255) # Yellow - -1 <= z-score <= 1 (near average)

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

    def _update_acceleration(self, track_id: int, speed_px_per_sec: float, frame_idx: int) -> float:
        """Update speed history and calculate acceleration in px/s² (0 if not enough data)."""
        tid = int(track_id)
        if tid not in self._speed_history:
            self._speed_history[tid] = []
        speed_hist = self._speed_history[tid]
        speed_hist.append((frame_idx, speed_px_per_sec))
        if len(speed_hist) > self._speed_history_max_len:
            speed_hist.pop(0)
        
        # Need at least 2 speed measurements to calculate acceleration
        if len(speed_hist) < 2 or self.fps <= 0:
            return 0.0
        
        (f0, v0), (f1, v1) = speed_hist[-2], speed_hist[-1]
        dt_frames = f1 - f0
        if dt_frames <= 0:
            return 0.0
        dt_sec = dt_frames / self.fps
        acceleration = (v1 - v0) / dt_sec  # Change in speed over time
        return float(acceleration)

    def _update_jerk(self, track_id: int, acceleration_px_per_sec2: float, frame_idx: int) -> float:
        """Update acceleration history and calculate jerk in px/s³ (0 if not enough data)."""
        tid = int(track_id)
        if tid not in self._acceleration_history:
            self._acceleration_history[tid] = []
        accel_hist = self._acceleration_history[tid]
        accel_hist.append((frame_idx, acceleration_px_per_sec2))
        if len(accel_hist) > self._acceleration_history_max_len:
            accel_hist.pop(0)
        
        # Need at least 2 acceleration measurements to calculate jerk
        if len(accel_hist) < 2 or self.fps <= 0:
            return 0.0
        
        (f0, a0), (f1, a1) = accel_hist[-2], accel_hist[-1]
        dt_frames = f1 - f0
        if dt_frames <= 0:
            return 0.0
        dt_sec = dt_frames / self.fps
        jerk = (a1 - a0) / dt_sec  # Change in acceleration over time
        return float(jerk)

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

    def _indicator_huggingface_model(self, frame: np.ndarray) -> tuple[bool, dict]:
        """Indicator: Hugging Face accident detection model.
        
        Args:
            frame: BGR frame from OpenCV
            
        Returns:
            (is_positive, details): Boolean indicating crash detected, and details dict
        """
        if not self.enable_huggingface_indicator or self.crash_pipe is None:
            return False, {'enabled': False}
        
        try:
            # Convert BGR to RGB for PIL
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Run inference
            results = self.crash_pipe(pil_image)
            
            # Results are typically a list of dicts with 'label' and 'score'
            if isinstance(results, list) and len(results) > 0:
                top_result = results[0] if isinstance(results[0], dict) else results
                label = top_result.get('label', '')
                confidence = top_result.get('score', 0.0)
                
                # Check if it's classified as an accident/crash
                is_crash = 'accident' in label.lower() or 'crash' in label.lower()
                is_positive = is_crash and confidence >= 0.5
                
                return confidence, {
                    'label': label,
                    'confidence': confidence,
                    'threshold_met': is_positive
                }
        except Exception as e:
            print(f"Error in Hugging Face indicator: {e}")
            return False, {'error': str(e)}
        
        return False, {'no_results': True}
    
    def _indicator_sudden_stop(self, frame_idx: int) -> tuple[bool, dict]:
        """Indicator: Detect sudden stops using MLE Gaussian on grid cell acceleration history.
        Fires when deceleration has high negative z-score (significantly below mean).
        
        Args:
            frame_idx: Current frame index
            
        Returns:
            (is_positive, details): Boolean indicating crash detected, and details dict
        """
        min_datapoints = 5  # Minimum datapoints per grid cell for reliable Gaussian estimate
        min_cells = 10 # Need at least 2 grid cells with sudden stops
        z_score_threshold = -2.0  # Z-score threshold (negative = deceleration below mean)
        
        cells_with_sudden_stop = 0
        max_z_score = 0.0
        max_deceleration = 0.0
        cells_checked = 0
        cells_insufficient_data = 0
        
        # Check grid cells for acceleration values
        for grid_cell, accelerations in self._acceleration_database.items():
            if not accelerations:
                continue
            
            cells_checked += 1
            
            # Only consider grid cells with sufficient datapoints
            if len(accelerations) < min_datapoints:
                cells_insufficient_data += 1
                continue
            
            # Extract acceleration values (excluding the latest for historical mean)
            latest_accel = accelerations[-1]
            historical_accels = accelerations[:-1]  # Use all but latest for distribution
            
            # MLE Gaussian estimation
            mean = np.mean(historical_accels)
            variance = np.mean((historical_accels - mean) ** 2)
            std_dev = np.sqrt(variance)
            
            # Handle edge case where std_dev is 0
            if std_dev == 0:
                # If all historical accelerations are the same, compare directly
                if latest_accel < mean:
                    z_score = -1.0
                else:
                    continue
            else:
                # Calculate z-score
                z_score = (latest_accel - mean) / std_dev
            
            # Check if z-score indicates sudden stop (negative and high magnitude)
            if z_score <= z_score_threshold:
                cells_with_sudden_stop += 1
                max_z_score = min(max_z_score, z_score)
                max_deceleration = min(max_deceleration, latest_accel)
        
        # Only return positive if we have enough reliable data and grid cells
        is_positive = cells_with_sudden_stop >= min_cells and cells_checked > 0
        
        return is_positive, {
            'cells_with_sudden_stop': cells_with_sudden_stop,
            'max_z_score': max_z_score,
            'max_deceleration': max_deceleration,
            'z_score_threshold': z_score_threshold,
            'cells_checked': cells_checked,
            'cells_insufficient_data': cells_insufficient_data,
            'min_datapoints_required': min_datapoints
        }
    
    def _indicator_person_detected(self, result) -> tuple[bool, dict]:
        """Indicator: Detect if a person (class 0) is present in the frame.
        
        Args:
            result: YOLO detection result
            
        Returns:
            (is_positive, details): Boolean indicating person detected, and details dict
        """
        if result is None or result.boxes is None:
            return False, {'enabled': False}
        
        # Check if any detected objects are class 0 (person)
        if result.boxes.cls is not None:
            classes = result.boxes.cls.cpu().numpy()
            person_indices = np.where(classes == 0)[0]
            
            if len(person_indices) > 0:
                # Get confidence scores for detected persons
                if result.boxes.conf is not None:
                    confidences = result.boxes.conf.cpu().numpy()
                    person_confidences = confidences[person_indices]
                    max_confidence = float(np.max(person_confidences))
                    avg_confidence = float(np.mean(person_confidences))
                else:
                    max_confidence = 1.0
                    avg_confidence = 1.0
                
                return True, {
                    'person_count': len(person_indices),
                    'max_confidence': max_confidence,
                    'avg_confidence': avg_confidence
                }
        
        return False, {'person_count': 0}
    
    def _indicator_high_jerk(self, frame_idx: int) -> tuple[bool, dict]:
        """Indicator: Detect high jerk values using MLE Gaussian on jerk history.
        Fires when absolute jerk has high positive z-score (significantly above mean).
        
        Args:
            frame_idx: Current frame index
            
        Returns:
            (is_positive, details): Boolean indicating crash detected, and details dict
        """
        min_datapoints = 5  # Minimum datapoints per grid cell for reliable Gaussian estimate
        min_objects = 10  # Need at least 1 object with high jerk
        z_score_threshold = 2.0  # Z-score threshold (positive = absolute jerk above mean)
        
        objects_with_high_jerk = 0
        max_z_score = 0.0
        max_jerk = 0.0
        cells_checked = 0
        cells_insufficient_data = 0
        
        # Check grid cells for high jerk values
        for grid_cell, jerks in self._jerk_database.items():
            if not jerks:
                continue
            
            cells_checked += 1
            
            # Only consider grid cells with sufficient datapoints
            if len(jerks) < min_datapoints:
                cells_insufficient_data += 1
                continue
            
            # Use absolute jerk values for analysis
            abs_jerks = [abs(j) for j in jerks]
            latest_abs_jerk = abs_jerks[-1]
            historical_abs_jerks = abs_jerks[:-1]  # Use all but latest for distribution
            
            # MLE Gaussian estimation
            mean = np.mean(historical_abs_jerks)
            variance = np.mean((historical_abs_jerks - mean) ** 2)
            std_dev = np.sqrt(variance)
            
            # Handle edge case where std_dev is 0
            if std_dev == 0:
                # If all historical jerks are the same, compare directly
                if latest_abs_jerk > mean:
                    z_score = 1.0
                else:
                    continue
            else:
                # Calculate z-score
                z_score = (latest_abs_jerk - mean) / std_dev
            
            # Check if z-score indicates high jerk (positive and high magnitude)
            if z_score >= z_score_threshold:
                objects_with_high_jerk += 1
                max_z_score = max(max_z_score, z_score)
                max_jerk = max(max_jerk, latest_abs_jerk)
        
        # Only return positive if we have enough reliable data
        is_positive = objects_with_high_jerk >= min_objects and cells_checked > 0
        
        return is_positive, {
            'objects_with_high_jerk': objects_with_high_jerk,
            'max_z_score': max_z_score,
            'max_jerk': max_jerk,
            'z_score_threshold': z_score_threshold,
            'cells_checked': cells_checked,
            'cells_insufficient_data': cells_insufficient_data,
            'min_datapoints_required': min_datapoints
        }
    
    def crash_detected(self, frame: np.ndarray, frame_idx: int, result=None) -> tuple[bool, dict]:
        """Main crash detection function that checks multiple indicators.
        
        Args:
            frame: BGR frame from OpenCV
            frame_idx: Current frame index
            result: Optional YOLO detection result for person detection
            
        Returns:
            (is_crash, indicators): Boolean indicating crash detected, and dict of all indicator results
        """
        if not self.enable_crash_detection:
            return False, {}
    
        weights = {}
        weights['huggingface_model'] = 5
        weights['sudden_stop'] = 2
        weights['high_jerk'] = 2
        weights['person_detected'] = 5
        # Normalize using softmax (exp and normalize)
        total = sum(math.exp(v) for v in weights.values())
        weights = {k: math.exp(v) / total for k, v in weights.items()}
        indicators = {}
        confidence_score = 0.0
        
        # Check each indicator
        # 1. Hugging Face model indicator
        hf_result, hf_details = self._indicator_huggingface_model(frame)
        indicators['huggingface_model'] = {
            'positive': hf_result,
            'details': hf_details
        }
        if hf_result:
            confidence_score += weights['huggingface_model']
        
        # 2. Sudden stop indicator
        sudden_stop_result, sudden_stop_details = self._indicator_sudden_stop(frame_idx)
        indicators['sudden_stop'] = {
            'positive': sudden_stop_result,
            'details': sudden_stop_details
        }
        if sudden_stop_result:
            confidence_score += weights['sudden_stop']
        
        # 3. High jerk indicator
        high_jerk_result, high_jerk_details = self._indicator_high_jerk(frame_idx)
        indicators['high_jerk'] = {
            'positive': high_jerk_result,
            'details': high_jerk_details
        }
        if high_jerk_result:
            confidence_score += weights['high_jerk']
        
        # 4. Person detected indicator
        person_result, person_details = self._indicator_person_detected(result)
        indicators['person_detected'] = {
            'positive': person_result,
            'details': person_details
        }
        if person_result:
            confidence_score += weights['person_detected']
        
        # Debug output (uncomment to debug)
        # print(f"Debug - person_result: {person_result}, confidence_score: {confidence_score:.4f}, threshold: {self.crash_indicator_threshold:.4f}, weights: {weights}")
        
        # Determine if crash is detected based on threshold
        is_crash = confidence_score >= self.crash_indicator_threshold
        
        # Store results for debugging
        self.last_crash_indicators = indicators
        
        return is_crash, indicators

    def run(self):
        print("Click on any object to track")
        print("Press 'q' to quit")
        print("Press 'c' to clear selection")
        print("Press 'g' to toggle grid visualization")

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
                classes=[0, 2, 3, 5, 7],
                max_det=500,
                tracker="botsort.yaml",
                device="mps",
                verbose=False,  # Suppress YOLO output
            )
            self.ann = Annotator(frame, line_width=2, font_size=1)

            n_det = 0
            mean_conf = 0.0
            min_box_area = 0.0
            current_result = None  # Store result for crash detection
            if results and len(results) > 0:
                result = results[0]
                current_result = result
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
                            acceleration_px_s2 = self._update_acceleration(int(id), speed_px_s, frame_idx)
                            jerk_px_s3 = self._update_jerk(int(id), acceleration_px_s2, frame_idx)
                            
                            # Determine grid cell for this object
                            grid_cell = self._pixel_to_grid_cell(cx, cy)
                            
                            # Add speed to database indexed by grid cell (groups neighboring pixels)
                            # Keep only last 50 instances per grid cell (in-memory only)
                            # Add BEFORE calculating color so current measurement is included
                            if grid_cell not in self._speed_database:
                                self._speed_database[grid_cell] = []
                            self._speed_database[grid_cell].append(speed_px_s)
                            # Limit to last 50 instances
                            if len(self._speed_database[grid_cell]) > self._speed_database_max_len:
                                self._speed_database[grid_cell].pop(0)
                            
                            # Calculate color based on historical data (including current speed)
                            color_bgr = self._speed_to_color_bgr(speed_px_s, grid_cell)
                            
                            # Add acceleration to database indexed by grid cell
                            # Keep only last 50 instances per grid cell (in-memory only)
                            if grid_cell not in self._acceleration_database:
                                self._acceleration_database[grid_cell] = []
                            self._acceleration_database[grid_cell].append(acceleration_px_s2)
                            # Limit to last 50 instances
                            if len(self._acceleration_database[grid_cell]) > self._acceleration_database_max_len:
                                self._acceleration_database[grid_cell].pop(0)
                            
                            # Add jerk to database indexed by grid cell
                            # Keep only last 50 instances per grid cell (in-memory only)
                            if grid_cell not in self._jerk_database:
                                self._jerk_database[grid_cell] = []
                            self._jerk_database[grid_cell].append(jerk_px_s3)
                            # Limit to last 50 instances
                            if len(self._jerk_database[grid_cell]) > self._jerk_database_max_len:
                                self._jerk_database[grid_cell].pop(0)
                            
                            self.ann.box_label(box, f"{speed_px_s:.0f} px/s", color=color_bgr)

            # Crash detection (check periodically to reduce computation)
            crash_detected = False
            if self.enable_crash_detection and frame_idx % self.crash_check_interval == 0:
                crash_detected, indicators = self.crash_detected(frame, frame_idx, current_result)
                if crash_detected:
                    self.last_crash_result = {
                        'is_crash': True,
                        'frame_idx': frame_idx,
                        'indicators': indicators
                    }
                    # Print detailed indicator information
                    positive_indicators = [name for name, data in indicators.items() if data['positive']]
                    negative_indicators = [name for name, data in indicators.items() if not data['positive']]
                    
                    print(f"\n{'='*60}")
                    print(f"⚠️  CRASH DETECTED! Frame {frame_idx}")
                    print(f"{'='*60}")
                    print(f"Indicator Status: {len(positive_indicators)}/{len(indicators)} indicators positive")
                    print(f"\n✅ POSITIVE INDICATORS ({len(positive_indicators)}):")
                    for name in positive_indicators:
                        data = indicators[name]
                        details = data.get('details', {})
                        print(f"   • {name.replace('_', ' ').title()}")
                        # Print specific details for each indicator type
                        if name == 'huggingface_model' and 'label' in details:
                            print(f"     - Label: {details['label']}")
                            print(f"     - Confidence: {details.get('confidence', 0):.3f}")
                        elif name == 'sudden_stop' and 'cells_with_sudden_stop' in details:
                            print(f"     - Grid cells with sudden stop: {details['cells_with_sudden_stop']}")
                            print(f"     - Max z-score: {details.get('max_z_score', 0):.2f} (threshold: {details.get('z_score_threshold', 0):.1f})")
                            print(f"     - Max deceleration: {details.get('max_deceleration', 0):.1f} px/s²")
                            print(f"     - Grid cells checked: {details.get('cells_checked', 0)}")
                            print(f"     - Cells with insufficient data: {details.get('cells_insufficient_data', 0)} (min {details.get('min_datapoints_required', 0)} required)")
                        elif name == 'high_jerk' and 'objects_with_high_jerk' in details:
                            print(f"     - Grid cells with high jerk: {details['objects_with_high_jerk']}")
                            print(f"     - Max z-score: {details.get('max_z_score', 0):.2f} (threshold: {details.get('z_score_threshold', 0):.1f})")
                            print(f"     - Max jerk: {details.get('max_jerk', 0):.1f} px/s³")
                            print(f"     - Grid cells checked: {details.get('cells_checked', 0)}")
                            print(f"     - Cells with insufficient data: {details.get('cells_insufficient_data', 0)} (min {details.get('min_datapoints_required', 0)} required)")
                        elif name == 'person_detected' and 'person_count' in details:
                            print(f"     - Persons detected: {details['person_count']}")
                            if details.get('person_count', 0) > 0:
                                print(f"     - Max confidence: {details.get('max_confidence', 0):.3f}")
                                print(f"     - Avg confidence: {details.get('avg_confidence', 0):.3f}")
                    
                    if negative_indicators:
                        print(f"\n❌ NEGATIVE INDICATORS ({len(negative_indicators)}):")
                        for name in negative_indicators:
                            print(f"   • {name.replace('_', ' ').title()}")
                    print(f"{'='*60}\n")
                else:
                    self.last_crash_result = None
            
            # Display crash alert on frame if detected
            if crash_detected or (self.last_crash_result and self.last_crash_result.get('is_crash', False)):
                h, w = frame.shape[:2]
                
                # Get indicator summary
                if self.last_crash_result and 'indicators' in self.last_crash_result:
                    indicators = self.last_crash_result['indicators']
                    positive_count = sum(1 for ind in indicators.values() if ind['positive'])
                    total_count = len(indicators)
                    alert_text = f"CRASH DETECTED! ({positive_count}/{total_count} indicators)"
                else:
                    alert_text = "CRASH DETECTED!"
                
                text_size = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
                text_x = (w - text_size[0]) // 2
                text_y = 50
                # Draw background rectangle
                cv2.rectangle(frame, (text_x - 10, text_y - 35), (text_x + text_size[0] + 10, text_y + 10), (0, 0, 255), -1)
                # Draw text
                cv2.putText(frame, alert_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
                # Draw red border around entire frame
                cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 10)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_luminance = float(np.mean(gray))
            # print(f"frame {frame_idx}: n_det={n_det} mean_conf={mean_conf:.3f} min_box_area={min_box_area:.0f} mean_luminance={mean_luminance:.1f}")
            frame_idx += 1

            # Draw grid overlay if enabled
            self._draw_grid(frame)

            # Write frame to output video if writer is set
            if self.video_writer is not None:
                self.video_writer.write(frame)

            cv2.imshow("Object Tracking", frame)
            key = cv2.waitKey(self.delay) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                self.selected_id = None
                print("Selection cleared")
            elif key == ord("g"):
                self._show_grid = not self._show_grid
                print(f"Grid visualization: {'ON' if self._show_grid else 'OFF'}")

        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            print(f"Output video saved to: {self.output_video_path}")
        cv2.destroyAllWindows()
        
        # Save database to file
        self._save_database()

if __name__ == "__main__":
    import sys
    
    # Example usage:
    # For live stream/URL:
    #   python object_tracking.py
    # For video file:
    #   python object_tracking.py path/to/video.mp4
    #   python object_tracking.py path/to/video.mp4 output.mp4  # with output video
    
    if len(sys.argv) > 1:
        # Video file mode
        video_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        
        print(f"Processing video file: {video_file}")
        if output_file:
            print(f"Output will be saved to: {output_file}")
        
        tracker = ObjectTracker(
            model="yolo26s.pt",
            video_source=video_file,
            output_video=output_file,
        )
    else:
        # Live stream mode (default)
        tracker = ObjectTracker(
            model="yolo26m.pt",
            video_source="https://cdn3.wowza.com/1/T05XOENNUVZBQ0cr/STJqWUVl/hls/live/playlist.m3u8",
            # Alternative stream:
            # video_source="https://cdn3.wowza.com/1/SkRQeFhmUk9sTDJG/dkovKzdK/hls/live/playlist.m3u8",
        )
    
    tracker.run()