import os
import time
import json
import logging
from collections import defaultdict
from typing import List, Optional

import cv2
import numpy as np
from datetime import datetime
from path_utils import parse_datetime_from_path

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from person_id import PersonReID

try:
    from ultralytics import YOLO
except Exception as e:
    raise

# --- Import teslapy for Tesla integration ---
try:
    import teslapy
except ImportError:
    teslapy = None

logger = logging.getLogger()


class _TimingProfiler:
    def __init__(self, enabled: bool = False):
        self.enabled = bool(enabled)
        self.totals_s = defaultdict(float)
        self.counts = defaultdict(int)

    def add(self, key: str, delta_s: float, count: int = 1) -> None:
        if not self.enabled:
            return
        try:
            if delta_s < 0:
                return
        except Exception:
            return
        self.totals_s[key] += float(delta_s)
        self.counts[key] += int(count)

    def report_lines(self, *, title: str, wall_s: Optional[float] = None, top_n: int = 20) -> List[str]:
        if not self.enabled:
            return []

        items = [(k, self.totals_s.get(k, 0.0), self.counts.get(k, 0)) for k in self.totals_s.keys()]
        items.sort(key=lambda x: x[1], reverse=True)
        total_profiled = sum(v for _k, v, _c in items)

        lines = [f"{title}:" ]
        if wall_s is not None:
            lines.append(f"  wall={wall_s:.3f}s profiled_sum={total_profiled:.3f}s")
        else:
            lines.append(f"  profiled_sum={total_profiled:.3f}s")

        if total_profiled <= 0:
            return lines

        for k, v, c in items[: max(1, int(top_n))]:
            pct = 100.0 * (v / total_profiled) if total_profiled > 0 else 0.0
            per = (v / c) if c else 0.0
            lines.append(f"  - {k}: {v:.3f}s ({pct:.1f}%), n={c}, avg={per*1000.0:.2f}ms")

        if len(items) > top_n:
            rest = items[top_n:]
            rest_sum = sum(v for _k, v, _c in rest)
            pct_rest = 100.0 * (rest_sum / total_profiled) if total_profiled > 0 else 0.0
            lines.append(f"  - (other {len(rest)} stages): {rest_sum:.3f}s ({pct_rest:.1f}%)")

        return lines

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Motion detection configuration ---
ROI_CONFIG_FILE = os.path.join(SCRIPT_DIR, "roi.json")
ROI_CONFIG_FILE_1080P = os.path.join(SCRIPT_DIR, "roi-1080p.json")
ROI_CONFIG_FILE_4K = os.path.join(SCRIPT_DIR, "roi-4k.json")
PADDING_SECONDS = 1.5
WARMUP_FRAMES = 15
MAX_EVENT_GAP_SECONDS = 3.0
MIN_EVENT_DURATION_SECONDS = 0.8               # Events shorter than this are considered insignificant, but sent to Gemini for analysis
MIN_INSIGNIFICANT_EVENT_DURATION_SECONDS = 0.2 # Events shorter than this are discarded as noise/shadows
# Tracking/render speed-up thresholds
# <= TRACK_FULL_UNTIL_SECONDS: track all frames, render all frames
# > TRACK_FULL_UNTIL_SECONDS and <= TRACK_SKIP_FROM_SECONDS: track all frames, render every 2nd frame
# > TRACK_SKIP_FROM_SECONDS: track every 2nd frame, render every 2nd frame (legacy behavior)
TRACK_FULL_UNTIL_SECONDS = 6.0
TRACK_SKIP_FROM_SECONDS = 12.0
SAVE_INSIGNIFICANT_FRAMES = True
SEND_INSIGNIFICANT_FRAMES = False
PERSON_MIN_FRAMES = 10 # Minimum frames with person inside ROI to consider event significant, if less then sent to Gemini for analysis

# --- Tesla config ---
TESLA_EMAIL = os.getenv("TESLA_EMAIL")
TESLA_REFRESH_TOKEN = os.getenv("TESLA_REFRESH_TOKEN")
TESLA_SOC_FILE = os.path.join(SCRIPT_DIR, "temp", "tesla_soc.txt")
TESLA_LAST_CHECK = 0
TESLA_SOC_CHECK_ENABLED = bool(teslapy and TESLA_REFRESH_TOKEN and TESLA_EMAIL)

# --- Object Detection Configuration ---
OBJECT_DETECTION_MODEL_PATH = os.getenv("OBJECT_DETECTION_MODEL_PATH", default=os.path.join("models", "best_openvino_model"))
CONF_THRESHOLD = 0.5
DETECT_CLASSES = [0, 1]  # 0: person, 1: car
TRACK_ROI_ENABLED = True  # Enable tracker ROI crop (from roi.json: 'tracker_roi' or fallback to motion_detection_roi/legacy)
COLOR_PERSON = (100, 200, 0)
COLOR_CAR = (200, 120, 0)
COLOR_DEFAULT = (255, 255, 255)
COLOR_HIGHLIGHT = (80, 90, 245)
COLOR_LINE = (0, 255, 255)
HIGHLIGHT_WINDOW_FRAMES = 5 # Minimum highlight duration (frames) after entering tolerance band
STABLE_MIN_FRAMES = 2  # frames outside tolerance required to confirm stable side
DWELL_SECONDS = 2.0    # seconds to stay on the other side to confirm a crossing

# --- Person ReID Configuration ---
REID_ENABLED = True
REID_MODEL_PATH = os.path.join(
    SCRIPT_DIR,
    "models",
    "reid",
    "intel",
    "person-reidentification-retail-0288",
    "FP16",
    "person-reidentification-retail-0288.xml",
)
REID_GALLERY_PATH = os.getenv("REID_GALLERY_PATH", os.path.join(SCRIPT_DIR, "person_of_interest"))
REID_THRESHOLD = 0.6 # cosine similarity threshold
REID_SAMPLING_STRIDE = 2  # sample every Nth frame near the line for ReID
REID_MAX_SAMPLES = 128 # maximum number of ReID samples to collect per event
SAVE_REID_BEST_CROP = True
REID_TOP_K = 3  # save up to K best, diverse crops per event
REID_DIVERSITY_MIN_DIST = 0.2  # min cosine distance between selected embeddings
REID_NEGATIVE_GALLERY_PATH = os.getenv("REID_NEGATIVE_GALLERY_PATH", os.path.join(SCRIPT_DIR, "person_of_interest_negative"))
REID_NEGATIVE_MARGIN = 0.08  # match must exceed negatives by at least this cosine margin

# --- Load Object Detection Model ---
try:
    object_detection_model = YOLO(OBJECT_DETECTION_MODEL_PATH, task='detect')
    logger.info(f"Object detection model loaded successfully from {OBJECT_DETECTION_MODEL_PATH}.")
except Exception as e:
    logger.critical(f"Failed to load object detection model: {e}", exc_info=True)
    raise

# --- Resolution-aware overlay + ROI scaling configuration ---
# These values are applied at runtime based on input frame resolution.
# 1080p defaults preserve current behavior; 4K values scale visuals/tolerances.

# Overlay draw parameters (runtime-adjusted)
OVERLAY_FONT_SCALE = 1
OVERLAY_TEXT_THICKNESS = 1
OVERLAY_BOX_THICKNESS = 2
OVERLAY_LINE_THICKNESS = 1
OVERLAY_LABEL_BG_HEIGHT = 30
OVERLAY_PAD_X = 20
OVERLAY_PAD_Y = 14

# Per-resolution configuration (values that differ between 1080p and 4K)
RES_CONFIGS = {
    "1080p": {
        "LINE_Y": 810,
        "LINE_Y_TOLERANCE": 6,
        "CROP_PADDING": 30,
        "TRACK_ROI_PADDING": 10,
        "REID_LINE_EXTRA_TOLERANCE": 20,
        "REID_CROP_PADDING": 12,
        "MIN_CONTOUR_AREA": 1800,
        # Tesla hotspot (x, y) to identify car box
        "TESLA_HOTSPOT": (1150, 450),
        # Overlays
        "FONT_SCALE": 1,
        "TEXT_THICKNESS": 1,
        "BOX_THICKNESS": 2,
        "LINE_THICKNESS": 1,
        "LABEL_BG_HEIGHT": 30,
        "PAD_X": 20,
        "PAD_Y": 14,
        # Motion-only downscale factor
        "MOTION_DOWNSCALE": 1.0,
        # Motion scan stride (process every Nth frame in the motion detector only)
        "MOTION_STRIDE": 1,
        # Highlight output size (always 1080p)
        "OUTPUT_SIZE": (1920, 1080),
    },
    "4k": {
        # Approximate 2x scaling from 1080p
        "LINE_Y": 1600,
        "LINE_Y_TOLERANCE": 12,
        "CROP_PADDING": 60,
        "TRACK_ROI_PADDING": 20,
        "REID_LINE_EXTRA_TOLERANCE": 40,
        "REID_CROP_PADDING": 24,
        "MIN_CONTOUR_AREA": 7200,  # 4x area
        # Tesla hotspot (x, y) scaled for 4K framing
        "TESLA_HOTSPOT": (2300, 900),
        # Overlays (larger for readability)
        "FONT_SCALE": 2,
        "TEXT_THICKNESS": 2,
        "BOX_THICKNESS": 2,
        "LINE_THICKNESS": 2,
        "LABEL_BG_HEIGHT": 60,
        "PAD_X": 40,
        "PAD_Y": 28,
        # Motion-only downscale factor
        "MOTION_DOWNSCALE": 0.5,
        # Motion scan stride (process every Nth frame in the motion detector only)
        # This reduces CPU by skipping frame processing; tracking still runs at full fidelity on selected events.
        "MOTION_STRIDE": 2,
        # Highlight output size (always 1080p)
        "OUTPUT_SIZE": (1920, 1080),
    },
}

# Default Tesla hotspot (overridden by RES_CONFIGS at runtime)
TESLA_TX, TESLA_TY = 1150, 450

def select_pretraining_candidates(duration_secs: int, train_win_sec: int) -> list:
    candidates = []

    def add(start_sec: float):
        start = int(max(0, min(round(start_sec), max(0, duration_secs - train_win_sec))))
        for s in candidates:
            if not (start + train_win_sec <= s or s + train_win_sec <= start):
                return
        candidates.append(start)

    if duration_secs <= 12:
        add(duration_secs - train_win_sec)
    elif duration_secs <= 30:
        add(duration_secs - 10)
        add((duration_secs - train_win_sec) * 0.5)
    elif duration_secs <= 45:
        add(duration_secs - 10)
        add(0.6 * duration_secs)
    else:
        add(duration_secs - 15)
        add(duration_secs - 10)
        add(0.6 * duration_secs)

    if not candidates:
        for p in (0.5, 0.6, 0.75):
            add(p * duration_secs)
            if candidates:
                break
    return candidates

def scale_roi_points_if_needed(points: np.ndarray, frame_w: int, frame_h: int, file_basename: str) -> np.ndarray:
    """If ROI points are outside frame bounds (typical when 4K ROI used on 1080p),
    attempt to downscale by 0.5. If still out-of-bounds, clip and warn.
    """
    try:
        if points is None or len(points) == 0:
            return points
        max_x = int(np.max(points[:, 0]))
        max_y = int(np.max(points[:, 1]))
        if max_x <= frame_w and max_y <= frame_h:
            return points
        # Try 0.5 downscale (common 4K->1080p)
        scaled = points.astype(np.float32).copy()
        scaled[:, 0] *= 0.5
        scaled[:, 1] *= 0.5
        s_max_x = int(np.max(scaled[:, 0]))
        s_max_y = int(np.max(scaled[:, 1]))
        if s_max_x <= frame_w and s_max_y <= frame_h:
            logger.warning(f"[{file_basename}] ROI points exceed 1080p frame; downscaled ROI by 0.5.")
            return scaled.astype(np.int32)
        # Fallback: clip to frame bounds
        logger.warning(f"[{file_basename}] ROI points exceed frame and cannot be downscaled cleanly; clipping to bounds.")
        clipped = scaled
        clipped[:, 0] = np.clip(clipped[:, 0], 0, frame_w - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, frame_h - 1)
        return clipped.astype(np.int32)
    except Exception:
        return points


def scale_roi_points_to_frame(
    points: np.ndarray,
    frame_w: int,
    frame_h: int,
    file_basename: str,
    *,
    roi_name: str = "roi",
    log_level: int = logging.WARNING,
) -> np.ndarray:
    """Scales ROI points down to fit the current frame.

    Supports cases where ROI points are authored for 4K (3840x2160) or 1080p (1920x1080)
    but the input clip is smaller (e.g., 640x360). If points already fit, returns them.

    This is used for the low-res fallback pipeline and does not affect 4K/1080p paths.
    """
    try:
        if points is None or len(points) == 0:
            return points
        max_x = int(np.max(points[:, 0]))
        max_y = int(np.max(points[:, 1]))
        if max_x <= frame_w and max_y <= frame_h:
            return points

        # Try 0.5 downscale (common 4K->1080p), but DO NOT clip-to-bounds here.
        # Clipping would collapse the polygon for very small frames (e.g., 640x360).
        try:
            half = points.astype(np.float32).copy()
            half[:, 0] *= 0.5
            half[:, 1] *= 0.5
            h_max_x = int(np.max(half[:, 0]))
            h_max_y = int(np.max(half[:, 1]))
            if h_max_x <= frame_w and h_max_y <= frame_h:
                logger.warning(f"[{file_basename}] ROI points exceed frame; downscaled ROI by 0.5 to fit.")
                return half.astype(np.int32)
        except Exception:
            pass

        # Determine likely ROI base resolution by inspecting coordinate magnitudes.
        # Keep base dims consistent (avoid mixed 1920x2160).
        looks_4k = bool(max_x > 2500 or max_y > 1400)
        base_w = 3840 if looks_4k else 1920
        base_h = 2160 if looks_4k else 1080
        sx = float(frame_w) / float(base_w)
        sy = float(frame_h) / float(base_h)
        scaled = points.astype(np.float32).copy()
        scaled[:, 0] *= sx
        scaled[:, 1] *= sy

        # Clip to bounds as a final safety.
        scaled[:, 0] = np.clip(scaled[:, 0], 0, frame_w - 1)
        scaled[:, 1] = np.clip(scaled[:, 1], 0, frame_h - 1)
        try:
            logger.log(
                int(log_level),
                f"[{file_basename}] {roi_name} points scaled to fit low-res frame ({base_w}x{base_h} -> {frame_w}x{frame_h}, sx={sx:.3f}, sy={sy:.3f}).",
            )
        except Exception:
            logger.warning(
                f"[{file_basename}] {roi_name} points scaled to fit low-res frame ({base_w}x{base_h} -> {frame_w}x{frame_h}, sx={sx:.3f}, sy={sy:.3f})."
            )
        return scaled.astype(np.int32)
    except Exception:
        return points


def load_tesla_soc(filepath):
    """Loads the Tesla SoC from the cache file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            soc = int(f.read().strip())
            return soc
    return None


def check_tesla_soc(file_basename):
    """
    Checks the Tesla's state of charge (SoC), using a cached value if recent.
    
    1. Checks for `tesla_soc.txt`.
    2. If it exists and was modified < 2 hours ago, returns the value from the file.
    3. Otherwise, fetches the SoC from the Tesla API, saves it to the file, and returns it.

    Args:
        file_basename (str): The basename of the video file being processed, for logging.

    Returns:
        int: The battery level (0-100) or None if an error occurs.
    """
    global TESLA_LAST_CHECK
    soc = None
    if os.path.exists(TESLA_SOC_FILE):
        last_modified_time = os.path.getmtime(TESLA_SOC_FILE)
        soc = load_tesla_soc(TESLA_SOC_FILE)
        current_time = time.time()
        if (current_time - last_modified_time < 7200) or (current_time - TESLA_LAST_CHECK < 600):
            if soc is not None:
                logger.info(f"[{file_basename}] Using cached Tesla SoC: {soc}%")
                return soc
            else:
                logger.warning(f"[{file_basename}] Could not read cached SoC file. Will fetch fresh data.")

    logger.info(f"[{file_basename}] Fetching fresh Tesla SoC from API...")
    try:
        with teslapy.Tesla(email=TESLA_EMAIL, cache_file=os.path.join(SCRIPT_DIR, "temp", "tesla_token_cache.json")) as tesla:
            if not tesla.authorized:
                tesla.refresh_token(refresh_token=TESLA_REFRESH_TOKEN)
            vehicles = tesla.vehicle_list()
            if not vehicles:
                logger.error(f"[{file_basename}] No vehicles found in Tesla account.")
                return None
            vehicle = vehicles[0]
            charge_state = vehicle.get_vehicle_data().get('charge_state', {})
            battery_level = charge_state.get('battery_level')

            TESLA_LAST_CHECK = time.time()

            if battery_level is not None:
                logger.info(f"[{file_basename}] Successfully fetched Tesla SoC: {battery_level}%")
                try:
                    with open(TESLA_SOC_FILE, 'w') as f:
                        f.write(str(battery_level))
                except IOError as e:
                    logger.error(f"[{file_basename}] Could not write to Tesla SoC cache file: {e}")
                return battery_level
            else:
                if soc is not None:
                    logger.warning(f"[{file_basename}] Could not retrieve 'battery_level' from Tesla API response. Using cached Tesla SoC: {soc}%")
                    return soc
                else:
                    logger.warning(f"[{file_basename}] Could not retrieve 'battery_level' from Tesla API response and no cached SoC available.")
                    return None

    except Exception as e:
        TESLA_LAST_CHECK = time.time()
        if soc is not None:
            if "408 Client Error" in str(e):
                logger.info(f"[{file_basename}] Tesla is asleep. Using cached Tesla SoC: {soc}%", exc_info=False)
            else:
                logger.warning(f"[{file_basename}] Tesla call failed: {e}. Using cached Tesla SoC: {soc}%", exc_info=False)
            return soc
        else:
            logger.warning(f"[{file_basename}] Tesla call failed and no cached SoC available: {e}", exc_info=False)
            return None


def read_roi_config(file_path):
    """
    Reads and returns the raw content of roi.json.

    Returns:
        dict | list | None: Parsed JSON (dict for multi-ROI config, or list for legacy array of points), or None if missing/invalid.
    """
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def as_np_points(points):
    try:
        if points is None:
            return None
        return np.array(points, dtype=np.int32)
    except Exception:
        return None


def draw_tracked_box(frame, box, local_id, label_name, conf, soc, highlight=False, display_id=None):
    """
    Draws a bounding box with a label for a tracked object on a video frame.
    If the object is a car in a specific location, it may display Tesla SoC.

    Args:
        frame (np.ndarray): The video frame to draw on.
        box (list): The bounding box coordinates [x1, y1, x2, y2].
        local_id (int): The local ID assigned to the tracked object.
        label_name (str): The class name of the object (e.g., 'person', 'car').
        conf (float): The detection confidence score.
        soc (int | None): The Tesla State of Charge, if available.
    """
    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

    if highlight:
        color = COLOR_HIGHLIGHT
    else:
        if label_name == 'person':
            color = COLOR_PERSON
        elif label_name == 'car':
            color = COLOR_CAR
        else:
            color = COLOR_DEFAULT

    if label_name == 'person' and display_id is not None:
        # Show only session-wide person display ID; omit local/tracker IDs
        label_text = f"p{display_id} {conf:.0%}"
    else:
        label_text = f"{label_name} {local_id} {conf:.0%}"

    if TESLA_SOC_CHECK_ENABLED:
        if label_name == 'car':
            if x1 <= TESLA_TX <= x2 and y1 <= TESLA_TY <= y2:
                if soc is not None:
                    label_text = f"Tesla {conf:.0%} / SoC {soc}%"
                else:
                    label_text = f"Tesla {conf:.0%}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, OVERLAY_BOX_THICKNESS)

    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, OVERLAY_FONT_SCALE, OVERLAY_TEXT_THICKNESS)

    cv2.rectangle(frame, (x1, y1 - OVERLAY_LABEL_BG_HEIGHT), (x1 + w, y1), color, -1)
    cv2.putText(frame, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_DUPLEX, OVERLAY_FONT_SCALE, (255, 255, 255), OVERLAY_TEXT_THICKNESS, cv2.LINE_AA)


def format_mmss(seconds):
    """Formats an integer number of seconds as MM:SS (e.g., 00:53)."""
    try:
        secs = int(seconds)
    except Exception:
        secs = 0
    mins = secs // 60
    rem = secs % 60
    return f"{mins:02d}:{rem:02d}"


def draw_event_overlay(frame, event_idx, total_events, seconds_from_start):
    """
    Draws a static black box in the bottom-right with white text showing
    current event number, total events, and seconds from start of source video.

    Example: "1/2 - 00:02"

    Args:
        frame (np.ndarray): BGR frame to draw on.
        event_idx (int): Current event index (1-based).
        total_events (int): Total number of significant events.
        seconds_from_start (int): Seconds from the start of the source video.
    """
    h, w = frame.shape[:2]
    text = f"{event_idx}/{total_events} - {format_mmss(seconds_from_start)}"

    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = OVERLAY_FONT_SCALE
    thickness = OVERLAY_TEXT_THICKNESS
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x = OVERLAY_PAD_X
    pad_y = OVERLAY_PAD_Y

    # Bottom-right anchor with margins
    x2 = w
    y2 = h
    x1 = x2 - (text_w + pad_x * 2)
    y1 = y2 - (text_h + pad_y * 2)

    # Black box background
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

    # Text in white centered within padding
    text_x = x1 + pad_x
    text_y = y1 + pad_y + text_h - 2
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def detect_draw_and_save_snapshot(frame, soc, output_dir, input_video_path, file_basename, mid_frame_index, tag, classes=None):
    """
    Runs a single-frame detection (no tracking), draws boxes and overlays, and saves JPEG.

    Args:
        frame (np.ndarray): BGR frame to analyze and draw on.
        soc (int | None): Tesla SoC for labeling if applicable.
        output_dir (str): Base output directory to save the snapshot under a daily subfolder.
        input_video_path (str): Full input path to derive camera/time prefix.
        file_basename (str): Basename of the input file for logging/prefix.
        mid_frame_index (int): Frame index used for naming the snapshot.
        tag (str): Label for the snapshot type, e.g., 'insignificant' or 'no_person'.
        classes (list[int] | None): Optional class indices to detect (Ultralytics). Defaults to DETECT_CLASSES.

    Returns:
        tuple[str | None, int]: (saved path or None on failure, number of detections drawn)
    """
    if classes is None:
        classes = DETECT_CLASSES
    try:
        results = object_detection_model.predict(frame, imgsz=640, conf=CONF_THRESHOLD, verbose=False, classes=classes)
    except Exception as e:
        logger.warning("[%s] Snapshot detection failed: %s", file_basename, e)
        return None

    try:
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()
            class_names = object_detection_model.names

            local_id_counter = 1
            for box, cls, conf in zip(boxes, clss, confs):
                label_name = class_names[cls]
                draw_tracked_box(frame, box, local_id_counter, label_name, conf, soc, highlight=False, display_id=None)
                local_id_counter += 1

        # Draw line overlay similar to main path (use resolution-aware sizes)
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), COLOR_LINE, OVERLAY_LINE_THICKNESS)
        cv2.putText(frame, f"Hvirtka Y={LINE_Y}", (10, LINE_Y - 10),
                cv2.FONT_HERSHEY_DUPLEX, OVERLAY_FONT_SCALE, COLOR_LINE, OVERLAY_TEXT_THICKNESS, cv2.LINE_AA)

        # Save to daily folder inside output dir (YYYYMMDD)
        date_folder = datetime.now().strftime("%Y%m%d")
        daily_dir = os.path.join(output_dir, date_folder)
        os.makedirs(daily_dir, exist_ok=True)
        dt = parse_datetime_from_path(input_video_path)
        hour = dt.strftime('%H') if dt else (input_video_path.split(os.path.sep)[-2][-2:] if len(input_video_path.split(os.path.sep)) >= 2 else "")
        frame_filename = f"{hour}H{os.path.splitext(file_basename)[0]}_{tag}_{mid_frame_index}.jpg"
        frame_path = os.path.join(daily_dir, frame_filename)
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.info("[%s] Saved %s frame to %s", file_basename, tag, frame_path)
        return frame_path
    except Exception as e:
        logger.warning("[%s] Failed to save %s frame: %s", file_basename, tag, e)
        return None


# --- Lightweight geometry helpers for entity matching ---
def box_area(b):
    return max(1.0, float(max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])))


def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    areaA = box_area(a)
    areaB = box_area(b)
    return inter / (areaA + areaB - inter)


def center_distance(a, b):
    ax = (a[0] + a[2]) * 0.5
    ay = (a[1] + a[3]) * 0.5
    bx = (b[0] + b[2]) * 0.5
    by = (b[1] + b[3]) * 0.5
    dx = ax - bx
    dy = ay - by
    return (dx * dx + dy * dy) ** 0.5


def bbox_diag(b):
    w = max(0.0, b[2] - b[0])
    h = max(0.0, b[3] - b[1])
    return (w * w + h * h) ** 0.5


def detect_motion(input_video_path, output_dir):
    """
    Analyzes a video file for motion, identifies significant events, and uses an object tracker.

    This function performs several steps:
    1.  Crops the video to a region around the ROI for efficiency.
    2.  Uses a background subtractor to find frames with motion.
    3.  Groups motion frames into events and filters out short/insignificant ones.
    4.  For significant events, it runs a YOLO object tracker to identify and count objects.
    5.  It checks for 'gate crossing' events where people cross a predefined line.
    6.  Generates a highlight clip (.mp4) of significant events with tracked objects boxed.
    7.  Extracts and saves single frames (.jpg) for insignificant motion events.

    Args:
        input_video_path (str): The path to the input .mp4 video file.
        output_dir (str): The directory to save generated clips and frames.

    Returns:
        dict: A dictionary containing the analysis status ('significant_motion', 'no_motion', etc.),
              path to the generated clip, paths to insignificant frames, and detected object counts.
    """
    file_basename = os.path.basename(input_video_path)

    # Open video first to detect resolution and choose ROI config accordingly
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.error(f"[{file_basename}] Could not open video file {input_video_path}")
        return {'status': 'error', 'clip_path': None, 'insignificant_frames': []}

    profile_env = os.getenv("ANALYZE_VIDEO_PROFILE", "0").strip().lower()
    profile_enabled = profile_env not in ("", "0", "false", "no", "off")
    prof = _TimingProfiler(enabled=profile_enabled)
    wall_start = time.perf_counter()

    def _maybe_log_profile(status: str) -> None:
        if not prof.enabled:
            return
        wall_s = time.perf_counter() - wall_start
        title = f"[{file_basename}] PROFILE status={status}"
        for line in prof.report_lines(title=title, wall_s=wall_s, top_n=25):
            logger.info(line)

    start_time = time.time()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    supported_light_resolutions = {(640, 360), (896, 512)}
    low_res_mode = (orig_w, orig_h) in supported_light_resolutions

    is_4k = bool(orig_w >= 3800 and orig_h >= 2100)
    is_1080p = bool(orig_w >= 1900 and orig_h >= 1000)

    # Skip any unexpected resolution (camera should only produce 4K, 640x360, 896x512; 1080p kept for legacy).
    if not (is_4k or is_1080p or low_res_mode):
        logger.info(f"[{file_basename}] Unsupported clip resolution: {orig_w}x{orig_h}. Skipping analysis.")
        cap.release()
        _maybe_log_profile("no_motion_unsupported_res")
        return {
            'status': 'no_motion',
            'clip_path': None,
            'insignificant_frames': [],
            'low_res_clip': True,
            'frame_size': (orig_w, orig_h),
        }

    # Select resolution config (4K/1080p unchanged).
    res_key = "4k" if is_4k else "1080p"
    cfg = RES_CONFIGS[res_key]
    # Apply runtime config to globals (keep existing logic paths)
    global LINE_Y, LINE_Y_TOLERANCE, CROP_PADDING, TRACK_ROI_PADDING, REID_LINE_EXTRA_TOLERANCE, REID_CROP_PADDING, MIN_CONTOUR_AREA
    LINE_Y = cfg["LINE_Y"]
    LINE_Y_TOLERANCE = cfg["LINE_Y_TOLERANCE"]
    CROP_PADDING = cfg["CROP_PADDING"]
    TRACK_ROI_PADDING = cfg["TRACK_ROI_PADDING"]
    REID_LINE_EXTRA_TOLERANCE = cfg["REID_LINE_EXTRA_TOLERANCE"]
    REID_CROP_PADDING = cfg["REID_CROP_PADDING"]
    MIN_CONTOUR_AREA = cfg["MIN_CONTOUR_AREA"]
    # Overlays
    global OVERLAY_FONT_SCALE, OVERLAY_TEXT_THICKNESS, OVERLAY_BOX_THICKNESS, OVERLAY_LINE_THICKNESS, OVERLAY_LABEL_BG_HEIGHT, OVERLAY_PAD_X, OVERLAY_PAD_Y
    OVERLAY_FONT_SCALE = cfg["FONT_SCALE"]
    OVERLAY_TEXT_THICKNESS = cfg["TEXT_THICKNESS"]
    OVERLAY_BOX_THICKNESS = cfg["BOX_THICKNESS"]
    OVERLAY_LINE_THICKNESS = cfg["LINE_THICKNESS"]
    OVERLAY_LABEL_BG_HEIGHT = cfg["LABEL_BG_HEIGHT"]
    OVERLAY_PAD_X = cfg["PAD_X"]
    OVERLAY_PAD_Y = cfg["PAD_Y"]
    # Tesla hotspot coordinates
    global TESLA_TX, TESLA_TY
    TESLA_TX, TESLA_TY = cfg.get("TESLA_HOTSPOT", (TESLA_TX, TESLA_TY))
    # Motion-only downscale factor for 4K (keeps tracking at full-res)
    md_scale = float(cfg.get("MOTION_DOWNSCALE", 1.0))
    md_resize_interpolation = cv2.INTER_LINEAR if is_4k else cv2.INTER_AREA
    # Motion scan stride for the background-subtraction stage only
    motion_stride = int(cfg.get("MOTION_STRIDE", 1))
    # Output highlight clip size (always 1080p)
    output_size = cfg["OUTPUT_SIZE"]

    if low_res_mode:
        # Scale motion threshold and padding to low-res pixel space.
        # Baseline is 4K ROI/thresholds because the source camera is 4K.
        try:
            area_ratio = float(orig_w * orig_h) / float(3840 * 2160)
        except Exception:
            area_ratio = 0.1
        if area_ratio <= 0:
            area_ratio = 0.1
        # Extra sensitivity multiplier for low-res: morphology/thresholding tends to shrink blobs.
        MIN_CONTOUR_AREA = max(60, int(RES_CONFIGS["4k"]["MIN_CONTOUR_AREA"] * area_ratio * 0.5))
        CROP_PADDING = max(6, int(RES_CONFIGS["4k"]["CROP_PADDING"] * (float(orig_w) / 3840.0)))
        md_scale = 1.0
        motion_stride = 1
        output_size = (orig_w, orig_h)

    # Optional profiling/knobs
    # - MOTION_PRETRAIN_SECONDS=5 (default). Set to 0 to disable pretraining.
    # - MOTION_PRETRAIN_MODE=always|auto|never (default: auto)
    #     always: keep legacy behavior (best accuracy, more work)
    #     auto: only pretrain if start-of-clip appears "active" (saves time on no-motion clips)
    #     never: skip pretraining entirely
    # - MOTION_START_ACTIVITY_CHECK_SECONDS=1.0 (only used in auto mode)
    # - MOTION_START_ACTIVITY_THRESHOLD=0.02 (fraction of ROI pixels changing between frames)
    try:
        motion_pretrain_seconds = float(os.getenv("MOTION_PRETRAIN_SECONDS", "5").strip() or "5")
    except Exception:
        motion_pretrain_seconds = 5.0
    if motion_pretrain_seconds < 0:
        motion_pretrain_seconds = 0.0
    motion_pretrain_mode = os.getenv("MOTION_PRETRAIN_MODE", "auto").strip().lower() or "auto"
    if motion_pretrain_mode not in ("always", "auto", "never"):
        motion_pretrain_mode = "auto"
    try:
        start_check_seconds = float(os.getenv("MOTION_START_ACTIVITY_CHECK_SECONDS", "1.0").strip() or "1.0")
    except Exception:
        start_check_seconds = 1.0
    if start_check_seconds < 0:
        start_check_seconds = 0.0
    try:
        start_activity_thr = float(os.getenv("MOTION_START_ACTIVITY_THRESHOLD", "0.02").strip() or "0.02")
    except Exception:
        start_activity_thr = 0.02
    if start_activity_thr < 0:
        start_activity_thr = 0.0


    # Choose ROI config file based on resolution, fallback to default.
    # Low-res mode prefers the 4K ROI (and scales it down) because the source camera is 4K.
    if low_res_mode and os.path.exists(ROI_CONFIG_FILE_4K):
        roi_path = ROI_CONFIG_FILE_4K
    elif res_key == "4k" and os.path.exists(ROI_CONFIG_FILE_4K):
        roi_path = ROI_CONFIG_FILE_4K
    elif res_key == "1080p" and os.path.exists(ROI_CONFIG_FILE_1080P):
        roi_path = ROI_CONFIG_FILE_1080P
    else:
        roi_path = ROI_CONFIG_FILE
    roi_config = read_roi_config(roi_path)
    if roi_config is None:
        logger.error(f"[{file_basename}] ROI config file not found or invalid: {roi_path}.")
        _maybe_log_profile("error_roi_config")
        return {'status': 'error', 'clip_path': None, 'insignificant_frames': []}
    # Motion detection ROI: if dict, expect 'motion_detection_roi'; if legacy (list), use it directly
    if isinstance(roi_config, dict):
        roi_poly_points = as_np_points(roi_config.get('motion_detection_roi'))
        # Override line position from ROI config if provided
        try:
            if 'line_y' in roi_config and roi_config['line_y'] is not None:
                LINE_Y = int(roi_config['line_y'])
                logger.debug(f"[{file_basename}] LINE_Y loaded from ROI config: {LINE_Y}")
        except Exception:
            pass
    elif isinstance(roi_config, list):
        roi_poly_points = as_np_points(roi_config)
    else:
        roi_poly_points = None
    if roi_poly_points is None:
        logger.error(f"[{file_basename}] Motion detection ROI missing in {roi_path}.")
        _maybe_log_profile("error_roi_missing")
        return {'status': 'error', 'clip_path': None, 'insignificant_frames': []}
    # Fit ROI to the frame when necessary.
    if low_res_mode:
        roi_poly_points = scale_roi_points_to_frame(
            roi_poly_points,
            orig_w,
            orig_h,
            file_basename,
            roi_name="motion_detection_roi",
            log_level=logging.INFO,
        )
    elif res_key == "1080p":
        # If 1080p input and ROI points exceed frame, attempt to downscale ROI by 0.5
        roi_poly_points = scale_roi_points_if_needed(roi_poly_points, orig_w, orig_h, file_basename)
    logger.debug(f"[{file_basename}] Using ROI config: {roi_path}")

    x_coords = roi_poly_points[:, 0]
    y_coords = roi_poly_points[:, 1]
    crop_x1 = max(0, np.min(x_coords) - CROP_PADDING)
    crop_y1 = max(0, np.min(y_coords) - CROP_PADDING)
    crop_x2 = min(orig_w, np.max(x_coords) + CROP_PADDING)
    crop_y2 = min(orig_h, np.max(y_coords) + CROP_PADDING)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    local_roi_points = roi_poly_points.copy()
    local_roi_points[:, 0] -= crop_x1
    local_roi_points[:, 1] -= crop_y1

    analysis_roi_points = local_roi_points.astype(np.int32)

    # Optional: Load tracker-specific ROI (from roi.json) and compute a fixed crop bounding box (guarded by TRACK_ROI_ENABLED)
    track_roi_bbox = None
    if TRACK_ROI_ENABLED:
        # Load tracker ROI from roi.json; if missing or legacy, use motion_detection_roi
        if isinstance(roi_config, dict) and 'tracker_roi' in roi_config:
            track_roi_poly_points = as_np_points(roi_config.get('tracker_roi'))
        else:
            track_roi_poly_points = roi_poly_points
        if track_roi_poly_points is not None and len(track_roi_poly_points) >= 3:
            if low_res_mode:
                track_roi_poly_points = scale_roi_points_to_frame(
                    track_roi_poly_points,
                    orig_w,
                    orig_h,
                    file_basename,
                    roi_name="tracker_roi",
                    log_level=logging.DEBUG,
                )
            elif res_key == "1080p":
                track_roi_poly_points = scale_roi_points_if_needed(track_roi_poly_points, orig_w, orig_h, file_basename)
            tx_coords = track_roi_poly_points[:, 0]
            ty_coords = track_roi_poly_points[:, 1]
            tx1 = max(0, int(np.min(tx_coords) - TRACK_ROI_PADDING))
            ty1 = max(0, int(np.min(ty_coords) - TRACK_ROI_PADDING))
            tx2 = min(orig_w, int(np.max(tx_coords) + TRACK_ROI_PADDING))
            ty2 = min(orig_h, int(np.max(ty_coords) + TRACK_ROI_PADDING))
            if tx2 > tx1 and ty2 > ty1:
                track_roi_bbox = (tx1, ty1, tx2, ty2)
            # else invalid bbox -> leave None

    # Optional: Load a person-specific tracker ROI to filter person detections outside this polygon
    person_tracker_roi_points = None
    if isinstance(roi_config, dict):
        person_tracker_roi_points = as_np_points(roi_config.get('person_tracker_roi'))
    person_tracker_polygon_cv = None
    if person_tracker_roi_points is not None and len(person_tracker_roi_points) >= 3:
        try:
            if low_res_mode:
                person_tracker_roi_points = scale_roi_points_to_frame(
                    person_tracker_roi_points,
                    orig_w,
                    orig_h,
                    file_basename,
                    roi_name="person_tracker_roi",
                    log_level=logging.DEBUG,
                )
            elif res_key == "1080p":
                person_tracker_roi_points = scale_roi_points_if_needed(person_tracker_roi_points, orig_w, orig_h, file_basename)
            person_tracker_polygon_cv = person_tracker_roi_points.reshape((-1, 1, 2))
        except Exception:
            person_tracker_polygon_cv = None

    logger.info(
        f"[{file_basename}] Original frame: {orig_w}x{orig_h}. motion_detection_roi={crop_w}x{crop_h}, "
        f"tracker_roi={(f'{track_roi_bbox[2]-track_roi_bbox[0]}x{track_roi_bbox[3]-track_roi_bbox[1]}' if track_roi_bbox is not None else 'False')}, "
        f"person_tracker_roi={(f'{int(np.max(person_tracker_roi_points[:,0]) - np.min(person_tracker_roi_points[:,0]))}x{int(np.max(person_tracker_roi_points[:,1]) - np.min(person_tracker_roi_points[:,1]))}' if person_tracker_polygon_cv is not None else 'False')}, "
        f"motion_stride={motion_stride}"
    )

    # Low-res tends to need a bit more sensitivity.
    knn_history = 500
    knn_dist2 = 600
    if low_res_mode:
        knn_history = 300
        knn_dist2 = 400
    backSub = cv2.createBackgroundSubtractorKNN(history=knn_history, dist2Threshold=knn_dist2, detectShadows=True)
    roi_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [analysis_roi_points], 255)
    # Downscaled mask for motion detection (4K only)
    if md_scale != 1.0:
        small_w = max(1, int(round(crop_w * md_scale)))
        small_h = max(1, int(round(crop_h * md_scale)))
        roi_mask_small = cv2.resize(roi_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    else:
        roi_mask_small = roi_mask

    # Auto-mode: detect if clip starts with activity inside ROI.
    # This is a cheap heuristic to decide whether expensive tail-pretraining is necessary.
    start_is_active = True
    if motion_pretrain_mode == "auto" and start_check_seconds > 0:
        try:
            frames_to_check = int(round(min(max(0.0, start_check_seconds), 3.0) * fps))
            frames_to_check = max(2, frames_to_check)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            prev_gray = None
            active_hits = 0
            mask_area = int(cv2.countNonZero(roi_mask_small))
            for _i in range(frames_to_check):
                if prof.enabled:
                    t0 = time.perf_counter()
                    ret, frame = cap.read()
                    prof.add("startcheck.read", time.perf_counter() - t0)
                else:
                    ret, frame = cap.read()
                if not ret or frame is None:
                    break
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if prof.enabled:
                    t0 = time.perf_counter()
                if md_scale != 1.0:
                    cropped_small = cv2.resize(
                        cropped_frame,
                        (roi_mask_small.shape[1], roi_mask_small.shape[0]),
                        interpolation=md_resize_interpolation,
                    )
                else:
                    cropped_small = cropped_frame
                gray = cv2.cvtColor(cropped_small, cv2.COLOR_BGR2GRAY)
                gray = cv2.bitwise_and(gray, gray, mask=roi_mask_small)
                if prev_gray is not None and mask_area > 0:
                    diff = cv2.absdiff(gray, prev_gray)
                    _, diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                    change = int(cv2.countNonZero(diff))
                    frac = change / float(mask_area)
                    if frac >= start_activity_thr:
                        active_hits += 1
                prev_gray = gray
                if prof.enabled:
                    prof.add("startcheck.ops", time.perf_counter() - t0)
            # Consider start active if we had any meaningful change in consecutive frames.
            start_is_active = active_hits > 0
        except Exception:
            start_is_active = True
        finally:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    pre_trained = False
    pretrain_attempted = False
    # Define training window (in frames) before selecting candidates
    frames_to_train = int(round(motion_pretrain_seconds * fps))
    do_pretrain = frames_to_train > 0
    if motion_pretrain_mode == "never":
        do_pretrain = False
    elif motion_pretrain_mode == "auto":
        do_pretrain = do_pretrain and bool(start_is_active)

    if not do_pretrain:
        why = "disabled" if motion_pretrain_mode == "never" or frames_to_train <= 0 else "start looks static"
        logger.info(
            f"[{file_basename}] Motion pretraining skipped ({why}). "
            f"mode={motion_pretrain_mode}, seconds={motion_pretrain_seconds}, start_active={start_is_active}"
        )
        frames_to_train = 0
    else:
        logger.info(
            f"[{file_basename}] Motion pretraining enabled. mode={motion_pretrain_mode}, seconds={motion_pretrain_seconds}, start_active={start_is_active}"
        )
    # 4K: prefer tail-of-clip static intervals (end-15s, end-10s); otherwise percent-based.
    # 1080p: preserve legacy fixed candidates.
    if frames_to_train > 0 and res_key == "4k":
        duration_secs = max(0, int(total_frames / max(1.0, fps)))
        train_win_sec = max(1, int(round(frames_to_train / max(1.0, fps))))
        training_candidate_times = select_pretraining_candidates(duration_secs, train_win_sec)
        logger.info(f"[{file_basename}] Pretraining candidates (sec): {training_candidate_times}")
    elif frames_to_train > 0:
        training_candidate_times = [30, 40, 50, 20]
    else:
        training_candidate_times = []
    # Validate candidate segment across the entire training window to avoid
    # pretraining on periods with subtle motion.
    frames_to_sample = frames_to_train

    for start_sec in training_candidate_times:
        pretrain_attempted = True
        start_frame = int(start_sec * fps)
        if total_frames < start_frame + frames_to_sample:
            continue

        if prof.enabled:
            t0 = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            prof.add("pretrain.seek_candidate", time.perf_counter() - t0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        motion_detected_in_segment = False
        temp_backSub = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=400, detectShadows=True)

        for i in range(frames_to_sample):
            if prof.enabled:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                prof.add("pretrain.sample.read", time.perf_counter() - t0)
                if ret:
                    prof.counts["pretrain.sample.frames"] += 1
            else:
                ret, frame = cap.read()
            if not ret:
                break

            if prof.enabled:
                t_ops = time.perf_counter()
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            # Motion detection operates on downscaled ROI when md_scale != 1.0.
            # Optimize: resize first, then apply the (already-resized) ROI mask.
            if md_scale != 1.0:
                cropped_small = cv2.resize(
                    cropped_frame,
                    (roi_mask_small.shape[1], roi_mask_small.shape[0]),
                    interpolation=md_resize_interpolation,
                )
                roi_frame_small = cv2.bitwise_and(cropped_small, cropped_small, mask=roi_mask_small)
                fg_mask = temp_backSub.apply(roi_frame_small)
                mask_area = roi_mask_small.size
            else:
                roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
                fg_mask = temp_backSub.apply(roi_frame)
                mask_area = roi_mask.size

            if prof.enabled:
                prof.add("pretrain.sample.ops", time.perf_counter() - t_ops)

            if prof.enabled:
                t_cn = time.perf_counter()
                nz = cv2.countNonZero(fg_mask)
                prof.add("pretrain.sample.countNonZero", time.perf_counter() - t_cn)
            else:
                nz = cv2.countNonZero(fg_mask)

            if i > 5 and nz > (mask_area * 0.15):
                motion_detected_in_segment = True
                logger.info(f"[{file_basename}] Motion detected in pre-training candidate segment at {start_sec}s. Trying next segment.")
                break

        if not motion_detected_in_segment:
            logger.info(f"[{file_basename}] Found a static segment at {start_sec}s. Pre-training background model...")
            if prof.enabled:
                t0 = time.perf_counter()
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                prof.add("pretrain.seek_train", time.perf_counter() - t0)
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(frames_to_train):
                if prof.enabled:
                    t0 = time.perf_counter()
                    ret, frame = cap.read()
                    prof.add("pretrain.train.read", time.perf_counter() - t0)
                    if ret:
                        prof.counts["pretrain.train.frames"] += 1
                else:
                    ret, frame = cap.read()
                if not ret:
                    break

                if prof.enabled:
                    t_ops = time.perf_counter()
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if md_scale != 1.0:
                    cropped_small = cv2.resize(
                        cropped_frame,
                        (roi_mask_small.shape[1], roi_mask_small.shape[0]),
                        interpolation=md_resize_interpolation,
                    )
                    roi_frame_small = cv2.bitwise_and(cropped_small, cropped_small, mask=roi_mask_small)
                    backSub.apply(roi_frame_small)
                else:
                    roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
                    backSub.apply(roi_frame)

                if prof.enabled:
                    prof.add("pretrain.train.ops", time.perf_counter() - t_ops)

            pre_trained = True
            break

    if pre_trained:
        logger.info(f"[{file_basename}] Background model pre-trained. Starting main motion detection loop...")
        if prof.enabled:
            t0 = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            prof.add("motion.seek_start", time.perf_counter() - t0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        # If we didn't even attempt pretraining (auto/never), don't emit a misleading warning.
        if pretrain_attempted:
            logger.warning(f"[{file_basename}] Could not find a static segment to pre-train model. Using standard warm-up.")
        # Always reset to the beginning for the main motion scan.
        if prof.enabled:
            t0 = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            prof.add("motion.seek_start", time.perf_counter() - t0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    EFFECTIVE_WARMUP = 5 if pre_trained else WARMUP_FRAMES

    # Proper warmup: advance through the first frames while updating the subtractor,
    # but do NOT record motion events. This avoids spurious startup motion.
    if EFFECTIVE_WARMUP > 0:
        for _wi in range(int(EFFECTIVE_WARMUP)):
            if prof.enabled:
                t0 = time.perf_counter()
                ret, frame = cap.read()
                prof.add("motion.warmup.read", time.perf_counter() - t0)
            else:
                ret, frame = cap.read()
            if not ret or frame is None:
                break
            if prof.enabled:
                t0 = time.perf_counter()
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            if md_scale != 1.0:
                cropped_small = cv2.resize(
                    cropped_frame,
                    (roi_mask_small.shape[1], roi_mask_small.shape[0]),
                    interpolation=md_resize_interpolation,
                )
                roi_frame_small = cv2.bitwise_and(cropped_small, cropped_small, mask=roi_mask_small)
                backSub.apply(roi_frame_small)
            else:
                roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
                backSub.apply(roi_frame)
            if prof.enabled:
                prof.add("motion.warmup.ops", time.perf_counter() - t0)

    motion_events = []
    frame_index = int(EFFECTIVE_WARMUP)
    # At this point, warmup has already consumed EFFECTIVE_WARMUP frames and the decoder is positioned accordingly.
    # We now scan the remaining video sequentially, processing only every motion_stride-th frame.
    while frame_index < total_frames:
        if prof.enabled:
            t0 = time.perf_counter()
            grabbed = cap.grab()
            prof.add("motion.grab", time.perf_counter() - t0)
        else:
            grabbed = cap.grab()

        if not grabbed:
            break

        # Skip processing for stride, but still advance frame_index and decoder.
        if motion_stride > 1 and ((frame_index - EFFECTIVE_WARMUP) % motion_stride) != 0:
            if prof.enabled:
                prof.counts["motion.skipped_frames"] += 1
            frame_index += 1
            continue

        if prof.enabled:
            t0 = time.perf_counter()
            ret, frame = cap.retrieve()
            prof.add("motion.retrieve", time.perf_counter() - t0)
            if ret and frame is not None:
                prof.counts["motion.frames"] += 1
        else:
            ret, frame = cap.retrieve()

        if not ret or frame is None:
            frame_index += 1
            continue

        if prof.enabled:
            t_ops = time.perf_counter()
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if md_scale != 1.0:
            cropped_small = cv2.resize(
                cropped_frame,
                (roi_mask_small.shape[1], roi_mask_small.shape[0]),
                interpolation=md_resize_interpolation,
            )
            roi_frame_small = cv2.bitwise_and(cropped_small, cropped_small, mask=roi_mask_small)
            fg_mask = backSub.apply(roi_frame_small)
            # Kernel is in the downscaled pixel space.
            kernel_size = 3 if low_res_mode else 5
        else:
            roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
            fg_mask = backSub.apply(roi_frame)
            kernel_size = 3 if low_res_mode else 5

        if prof.enabled:
            prof.add("motion.roi+backsub", time.perf_counter() - t_ops)

        if prof.enabled:
            t_post = time.perf_counter()
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        open_iter = 1 if low_res_mode else 2
        dilate_iter = 2 if low_res_mode else 3
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=dilate_iter)
        if prof.enabled:
            prof.add("motion.morphology", time.perf_counter() - t_post)

        if prof.enabled:
            t_cnt = time.perf_counter()
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if prof.enabled:
            prof.add("motion.findContours", time.perf_counter() - t_cnt)

        # Effective area threshold accounts for motion downscale
        if prof.enabled:
            t_f = time.perf_counter()
        effective_min_area = float(MIN_CONTOUR_AREA) * (md_scale * md_scale)
        large_enough_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > effective_min_area]
        if prof.enabled:
            prof.add("motion.filterContours", time.perf_counter() - t_f)
        if large_enough_contours:
            motion_events.append((frame_index, large_enough_contours))

        frame_index += 1

    if not motion_events:
        logger.info(f"[{file_basename}] No significant motion found.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        _maybe_log_profile("no_motion")
        return {'status': 'no_motion', 'clip_path': None, 'insignificant_frames': []}

    motion_frame_indices = [item[0] for item in motion_events]

    max_gap_in_frames = MAX_EVENT_GAP_SECONDS * fps
    sub_clips = []
    if motion_frame_indices:
        clip_start = motion_frame_indices[0]
        clip_end = motion_frame_indices[0]
        for i in range(1, len(motion_frame_indices)):
            if motion_frame_indices[i] - clip_end > max_gap_in_frames:
                sub_clips.append((clip_start, clip_end))
                clip_start = motion_frame_indices[i]
            clip_end = motion_frame_indices[i]
        sub_clips.append((clip_start, clip_end))

    soc = check_tesla_soc(file_basename) if TESLA_SOC_CHECK_ENABLED else None

    logger.info(f"[{file_basename}] Found {len(sub_clips)} raw motion event(s). Filtering by duration...")
    significant_sub_clips = []
    insignificant_motion_frames = []
    all_shorter_than_insignificant = True
    for start_frame, end_frame in sub_clips:
        duration_frames = end_frame - start_frame
        duration_seconds = duration_frames / fps
        if duration_seconds >= MIN_EVENT_DURATION_SECONDS:
            logger.info(f"[{file_basename}]   - Event at {(start_frame / fps):.1f}s lasting {duration_seconds:.2f}s is SIGNIFICANT. Will process with tracker.")
            significant_sub_clips.append((start_frame, end_frame))
        elif duration_seconds >= MIN_INSIGNIFICANT_EVENT_DURATION_SECONDS:
            all_shorter_than_insignificant = False
            if SAVE_INSIGNIFICANT_FRAMES:
                logger.info(f"[{file_basename}]   - Event at {(start_frame / fps):.1f}s lasting {duration_seconds:.2f}s is insignificant. Extracting frame.")
                mid_frame_index = start_frame + (end_frame - start_frame) // 2

                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
                ret, frame = cap.read()
                if ret:
                    frame_path = detect_draw_and_save_snapshot(
                        frame=frame,
                        soc=soc,
                        output_dir=output_dir,
                        input_video_path=input_video_path,
                        file_basename=file_basename,
                        mid_frame_index=mid_frame_index,
                        tag="insignificant",
                        classes=DETECT_CLASSES,
                    )
                    if frame_path and SEND_INSIGNIFICANT_FRAMES:
                        insignificant_motion_frames.append(frame_path)
            else:
                logger.info(f"[{file_basename}]   - Event at {(start_frame / fps):.1f}s lasting {duration_seconds:.2f}s is insignificant. Skipping frame extraction.")
        else:
            logger.info(f"[{file_basename}]   - Event at {(start_frame / fps):.1f}s lasting {duration_seconds:.2f}s is too short. Discarding as noise/shadow.")

    # If all sub-clips were shorter than the insignificant threshold, treat as no motion
    if not significant_sub_clips and all_shorter_than_insignificant:
        logger.info(f"[{file_basename}] All motion events shorter than MIN_INSIGNIFICANT_EVENT_DURATION_SECONDS. Treating as no motion.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        _maybe_log_profile("no_motion_all_short")
        return {'status': 'no_motion', 'clip_path': None, 'insignificant_frames': []}

    if not significant_sub_clips:
        logger.info(f"[{file_basename}] No significant long-duration motion found.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        _maybe_log_profile("no_significant_motion")
        return {'status': 'no_significant_motion', 'clip_path': None, 'insignificant_frames': insignificant_motion_frames}

    output_filename = os.path.join(output_dir, file_basename)

    if low_res_mode:
        # Low-res lightweight path: no tracking, no ReID, no gate crossing.
        # Significant = motion-only rule (duration >= MIN_EVENT_DURATION_SECONDS).
        try:
            writer = None
            written_frame_count = 0
            logger.info(f"[{file_basename}] Low-res significant motion: generating highlight clip at source resolution {output_size[0]}x{output_size[1]} (no tracking).")

            for clip_index, (start_frame, end_frame) in enumerate(significant_sub_clips):
                padded_start = max(0, int(start_frame - fps * PADDING_SECONDS))
                padded_end = min(total_frames - 1, int(end_frame + fps * PADDING_SECONDS))
                cap.set(cv2.CAP_PROP_POS_FRAMES, padded_start)

                frame_idx = padded_start
                while frame_idx <= padded_end:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    # Optional minimal overlay: draw event index/time (no boxes).
                    try:
                        current_seconds = int((frame_idx - 1) / fps)
                        draw_event_overlay(frame, clip_index + 1, len(significant_sub_clips), current_seconds)
                    except Exception:
                        pass

                    if writer is None:
                        logger.info(
                            f"[{file_basename}] Initializing CRF-based H.264 writer (libx264, preset=faster, crf=28) at {output_size[0]}x{output_size[1]}..."
                        )
                        writer = FFMPEG_VideoWriter(
                            output_filename,
                            size=output_size,
                            fps=fps,
                            codec='libx264',
                            preset='faster',
                            threads=0,
                            ffmpeg_params=['-crf', '28', '-pix_fmt', 'yuv420p', '-movflags', '+faststart'],
                        )

                    if frame.shape[1] != output_size[0] or frame.shape[0] != output_size[1]:
                        # Safety: should not happen for low-res mode, but keep robust.
                        frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.write_frame(rgb)
                    written_frame_count += 1

                    frame_idx += 1

            cap.release()
            if writer is not None:
                logger.info(f"[{file_basename}] Finalizing video writer...")
                writer.close()

            elapsed_time = time.time() - start_time
            logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")

            if written_frame_count <= 0:
                _maybe_log_profile("low_res_no_frames")
                return {'status': 'no_motion', 'clip_path': None, 'insignificant_frames': insignificant_motion_frames, 'low_res_clip': True, 'frame_size': (orig_w, orig_h)}

            _maybe_log_profile("low_res_significant_motion")
            return {
                'status': 'significant_motion',
                'clip_path': output_filename,
                'insignificant_frames': insignificant_motion_frames,
                'low_res_clip': True,
                'frame_size': (orig_w, orig_h),
            }
        except Exception as e:
            logger.warning(f"[{file_basename}] Low-res highlight generation failed: {e}", exc_info=True)
            try:
                cap.release()
            except Exception:
                pass
            _maybe_log_profile("low_res_error")
            return {'status': 'error', 'clip_path': None, 'insignificant_frames': insignificant_motion_frames, 'low_res_clip': True, 'frame_size': (orig_w, orig_h)}

    logger.info(f"[{file_basename}] Starting object tracking for {len(significant_sub_clips)} significant event(s).")

    if hasattr(object_detection_model, 'predictor') and object_detection_model.predictor is not None:
        if hasattr(object_detection_model.predictor, 'trackers') and object_detection_model.predictor.trackers:
            object_detection_model.predictor.trackers[0].reset()
            logger.debug("[%s] Object tracker state reset.", file_basename)

    # Prepare ROI polygon for point-in-polygon checks
    roi_polygon_cv = roi_poly_points.reshape((-1, 1, 2))

    writer = None
    written_frame_count = 0
    class_names = object_detection_model.names
    id_mapping = {}
    class_counters = {}
    unique_objects_detected = {'person': set(), 'car': set()}
    persons_up = 0
    persons_down = 0
    # Session-wide display IDs for persons: local_id -> display_id
    person_display_ids = {}
    person_display_id_counter = 1
    session_display_initialized = False

    # Prepare accumulator for ReID samples across all kept/processed events
    reid_candidate_crops = []

    # Process each event separately and include only those with person inside ROI for >= PERSON_MIN_FRAMES
    for clip_index, (start_frame, end_frame) in enumerate(significant_sub_clips):
        duration_seconds = (end_frame - start_frame) / fps
        is_long_motion = duration_seconds > TRACK_FULL_UNTIL_SECONDS
        padding_seconds_adjusted = 1 if is_long_motion else PADDING_SECONDS
        padded_start = max(0, start_frame - int(padding_seconds_adjusted * fps))
        padded_end = min(total_frames, end_frame + int(padding_seconds_adjusted * fps))

        if clip_index == 0 and start_frame <= (EFFECTIVE_WARMUP + fps * 1.0):
            logger.info(f"[{file_basename}] {clip_index + 1} - Motion starts at the beginning. Including video from frame 0.")
            padded_start = 0

        # Decide per-event tracking and rendering stride
        if duration_seconds <= TRACK_FULL_UNTIL_SECONDS:
            tracker_stride = 1
            output_stride = 1
        elif duration_seconds <= TRACK_SKIP_FROM_SECONDS:
            tracker_stride = 1
            output_stride = 2
        else:
            tracker_stride = 2
            output_stride = 2

        if is_long_motion:
            logger.info(f"[{file_basename}] {clip_index + 1} - Long motion event ({duration_seconds:.2f}s). tracker_stride={tracker_stride}, output_stride={output_stride}.")

        # Event-scoped accumulators
        event_frames_rgb = []
        event_unique_objects = {'person': set(), 'car': set()}
        event_persons_up = 0
        event_persons_down = 0
        # Track per-person crossing state within the event to compute net result
        event_first_side = {}
        event_last_side = {}
        event_prev_y = {}
        event_cross_dirs = {}
        # Minimum highlight window per entity: starts when entering tolerance
        event_highlight_until = {}
        # Stable-side tracking and dwell confirmation per entity
        event_stable_side = {}
        event_stable_since = {}
        event_stable_frames = {}
        event_pending_cross = {}  # entity_id -> {'dir': 'up'|'down', 'since_frame': int}
        # Track if entity is inside line tolerance band to reduce log spam
        event_in_tolerance = {}
        # Event-local display IDs (used until first kept event seeds the session mapping)
        event_display_ids = {}
        # Event-entity display IDs (stable per physical person across tracker id flips)
        event_entity_display_ids = {}
        event_entity_display_id_counter = 1
        # Track last known side per tracker id to bridge gaps
        event_last_side_global = {}
        # Event-local entity matcher (stable IDs across tracker flips)
        event_entities = []  # list of dicts: {id, last_box, last_frame}
        event_entities_map = {}
        event_global_to_entity = {}
        event_next_entity_id = 1
        # Matcher thresholds
        IOU_STRONG = 0.5
        IOU_WEAK = 0.3
        CENTER_DIST_RATIO = 0.35
        SIZE_RATIO_LIMIT = 1.6
        MAX_GAP_FRAMES = int(2 * fps * tracker_stride)
        # Within-frame duplicate suppression for persons
        DUP_IOU = 0.7
        DUP_CENTER_RATIO = 0.25
        person_frames_in_roi = 0

        # Seek once to the start of the event, then advance sequentially to avoid decoder seek artifacts
        if prof.enabled:
            t0 = time.perf_counter()
            cap.set(cv2.CAP_PROP_POS_FRAMES, padded_start)
            prof.add("track.seek_event", time.perf_counter() - t0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, padded_start)
        frame_idx = padded_start
        while frame_idx < padded_end:
            if prof.enabled:
                t0 = time.perf_counter()
                grabbed = cap.grab()
                prof.add("track.grab", time.perf_counter() - t0)
            else:
                grabbed = cap.grab()
            if not grabbed:
                frame_idx += 1
                continue

            # Skip frames for tracking based on tracker_stride
            if (frame_idx - padded_start) % tracker_stride != 0:
                frame_idx += 1
                continue

            if prof.enabled:
                t0 = time.perf_counter()
                ret, frame = cap.retrieve()
                prof.add("track.retrieve", time.perf_counter() - t0)
                if ret and frame is not None:
                    prof.counts["track.frames"] += 1
            else:
                ret, frame = cap.retrieve()
            frame_idx += 1
            if not ret or frame is None:
                continue
            processed_offset = (frame_idx - 1) - padded_start
            needs_reid = (
                REID_ENABLED
                and REID_SAMPLING_STRIDE > 0
                and (processed_offset % REID_SAMPLING_STRIDE == 0)
            )
            frame_for_reid = None
            if needs_reid:
                # Keep a clean copy for ReID crops (avoid overlay artifacts)
                if prof.enabled:
                    t0 = time.perf_counter()
                    frame_for_reid = frame.copy()
                    prof.add("track.frame_copy", time.perf_counter() - t0)
                else:
                    frame_for_reid = frame.copy()

            # Apply tracker ROI crop if available
            if track_roi_bbox is not None:
                tx1, ty1, tx2, ty2 = track_roi_bbox
                track_frame = frame[ty1:ty2, tx1:tx2]
            else:
                track_frame = frame

            if prof.enabled:
                t0 = time.perf_counter()
                results = object_detection_model.track(
                    track_frame, imgsz=640, conf=CONF_THRESHOLD, persist=True, verbose=False,
                    tracker="tracker.yaml", classes=DETECT_CLASSES
                )
                prof.add("track.yolo_track", time.perf_counter() - t0)
            else:
                results = object_detection_model.track(
                    track_frame, imgsz=640, conf=CONF_THRESHOLD, persist=True, verbose=False,
                    tracker="tracker.yaml", classes=DETECT_CLASSES
                )

            # Count whether this frame contains a person within ROI and whether any crossing/highlight happens
            frame_has_person_in_roi = False
            frame_has_crossing = False
            frame_has_highlight = False
            # Ensure defined even if there are no tracker detections this frame
            accepted_persons = []

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().tolist()
                # Offset boxes back to global coordinates if we tracked on a crop
                if track_roi_bbox is not None:
                    tx1, ty1, tx2, ty2 = track_roi_bbox
                    for i in range(len(boxes)):
                        boxes[i][0] += tx1
                        boxes[i][1] += ty1
                        boxes[i][2] += tx1
                        boxes[i][3] += ty1
                global_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()
                # Sort detections by confidence (high->low) to keep best when suppressing duplicates
                dets = list(zip(boxes, global_ids, clss, confs))
                dets.sort(key=lambda d: d[3], reverse=True)
                # Prevent multiple detections mapping to same entity within a single frame
                frame_assigned_entities = set()
                # Track accepted person detections for duplicate suppression within this frame
                accepted_persons = []  # list of (box, entity_id)

                for box, global_id, cls, conf in dets:
                    label_name = class_names[cls]

                    # If configured, ignore person detections outside the person_tracker ROI
                    if label_name == 'person' and person_tracker_polygon_cv is not None:
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        if cv2.pointPolygonTest(person_tracker_polygon_cv, (cx, cy), False) < 0:
                            continue

                    if label_name not in class_counters:
                        class_counters[label_name] = 1
                    if global_id not in id_mapping:
                        id_mapping[global_id] = class_counters[label_name]
                        class_counters[label_name] += 1
                    local_id = id_mapping[global_id]

                    if label_name in event_unique_objects:
                        event_unique_objects[label_name].add(global_id)

                    # Line crossing tracking per person (compute net outcome per event)
                    if label_name == 'person':
                        # Suppress near-duplicate person detections within the same frame
                        dup_found = False
                        for acc_box, acc_eid in accepted_persons:
                            if iou(box, acc_box) >= DUP_IOU:
                                dup_found = True
                                break
                            # Also consider small center distance relative to size
                            dist_c = center_distance(box, acc_box)
                            diag_c = max(bbox_diag(acc_box), bbox_diag(box))
                            if diag_c > 0 and (dist_c / diag_c) <= DUP_CENTER_RATIO:
                                dup_found = True
                                break
                        if dup_found:
                            # Skip processing this duplicate detection
                            continue
                        # Assign stable entity_id
                        y_center = int((box[1] + box[3]) / 2)
                        current_side = 'above' if y_center < LINE_Y else 'below'
                        # 0) Prefer direct continuity by tracker id when available
                        accept = False
                        prev_eid = event_global_to_entity.get(global_id)
                        if prev_eid is not None and prev_eid in event_entities_map:
                            ent_prev = event_entities_map[prev_eid]
                            if (frame_idx - ent_prev['last_frame']) <= MAX_GAP_FRAMES and prev_eid not in frame_assigned_entities:
                                entity_id = prev_eid
                                accept = True

                        # 1) IoU-based matching if no direct continuity
                        best_id = None
                        best_ent = None
                        best_iou = 0.0
                        if not accept:
                            for ent in event_entities:
                                if (frame_idx - ent['last_frame']) > MAX_GAP_FRAMES:
                                    continue
                                iou_score = iou(box, ent['last_box'])
                                if iou_score > best_iou:
                                    best_iou = iou_score
                                    best_id = ent['id']
                                    best_ent = ent
                        if best_ent is not None:
                            diag = bbox_diag(best_ent['last_box'])
                            dist = center_distance(box, best_ent['last_box'])
                            area_now = box_area(box)
                            area_prev = box_area(best_ent['last_box'])
                            size_ratio = area_now / max(1e-6, area_prev)
                            if size_ratio < 1.0:
                                size_ratio = 1.0 / size_ratio
                            gap_frames = frame_idx - best_ent['last_frame']
                            gap_ratio = min(1.0, max(0.0, gap_frames / max(1.0, MAX_GAP_FRAMES)))
                            iou_strong_req = min(0.9, IOU_STRONG + 0.2 * gap_ratio)
                            iou_weak_req = min(0.8, IOU_WEAK + 0.2 * gap_ratio)
                            center_ratio_allowed = max(0.15, CENTER_DIST_RATIO * (1.0 - 0.5 * gap_ratio))
                            size_ratio_limit_eff = max(1.3, SIZE_RATIO_LIMIT - 0.3 * gap_ratio)
                            if (best_iou >= iou_strong_req) or (
                                best_iou >= iou_weak_req and dist <= center_ratio_allowed * max(1.0, diag) and size_ratio <= size_ratio_limit_eff
                            ):
                                if best_id not in frame_assigned_entities:
                                    entity_id = best_id
                                    accept = True
                        if not accept:
                            # Second-chance bridge: if recent entity is on opposite side, merge even with low IoU
                            fallback_id = None
                            for ent in event_entities:
                                if (frame_idx - ent['last_frame']) > MAX_GAP_FRAMES:
                                    continue
                                last_y = int((ent['last_box'][1] + ent['last_box'][3]) / 2)
                                last_side = 'above' if last_y < LINE_Y else 'below'
                                gap2 = frame_idx - ent['last_frame']
                                gap2_ratio = min(1.0, max(0.0, gap2 / max(1.0, MAX_GAP_FRAMES)))
                                dist2 = center_distance(box, ent['last_box'])
                                diag2 = bbox_diag(ent['last_box'])
                                # Allow tighter center distance as gap grows
                                allow_ratio = 0.6 * (1.0 - 0.5 * gap2_ratio)
                                # Prefer opposite side handoff, but for short gaps also allow same-side if very close
                                if last_side != current_side:
                                    if dist2 <= allow_ratio * max(1.0, diag2):
                                        fallback_id = ent['id']
                                        break
                                else:
                                    if gap2_ratio <= 0.3 and dist2 <= 0.4 * max(1.0, diag2):
                                        fallback_id = ent['id']
                                        break
                            if fallback_id is not None and fallback_id not in frame_assigned_entities:
                                entity_id = fallback_id
                                accept = True
                            else:
                                entity_id = event_next_entity_id
                                event_next_entity_id += 1
                                new_ent = {'id': entity_id, 'last_box': box[:], 'last_frame': frame_idx}
                                event_entities.append(new_ent)
                                event_entities_map[entity_id] = new_ent
                        else:
                            ent = event_entities_map[entity_id]
                            ent['last_box'] = box[:]
                            ent['last_frame'] = frame_idx
                        frame_assigned_entities.add(entity_id)
                        event_global_to_entity[global_id] = entity_id

                        # Determine display ID (tracker-based mapping with event-entity stability)
                        disp_id = None
                        if label_name == 'person':
                            if session_display_initialized:
                                if local_id not in person_display_ids:
                                    person_display_ids[local_id] = person_display_id_counter
                                    person_display_id_counter += 1
                                disp_id = person_display_ids[local_id]
                            else:
                                if entity_id not in event_entity_display_ids:
                                    event_entity_display_ids[entity_id] = event_entity_display_id_counter
                                    event_entity_display_id_counter += 1
                                disp_id = event_entity_display_ids[entity_id]
                                # Mirror into event_display_ids keyed by local_id so we can seed later
                                if local_id not in event_display_ids:
                                    event_display_ids[local_id] = disp_id

                        # Previous side seen for this tracker id (global)
                        prev_global_side = event_last_side_global.get(global_id)
                        if entity_id not in event_prev_y:
                            event_prev_y[entity_id] = y_center
                            event_cross_dirs[entity_id] = set()
                            # If re-acquired and side differs, infer crossing and set start side accordingly
                            if prev_global_side is not None and prev_global_side != current_side:
                                # Set start side to previous side to ensure net count registers
                                event_first_side[entity_id] = prev_global_side
                                event_last_side[entity_id] = current_side
                                # Record direction for completeness
                                if prev_global_side == 'below' and current_side == 'above':
                                    event_cross_dirs[entity_id].add('up')
                                elif prev_global_side == 'above' and current_side == 'below':
                                    event_cross_dirs[entity_id].add('down')
                                # Mark crossing for frame inclusion; highlight is based on tolerance or window
                                frame_has_crossing = True
                                # Non-person labels don't use disp_id
                                if label_name != 'person':
                                    disp_id = None
                                # Highlight if currently in tolerance or still within minimum highlight window
                                in_tol_now = abs(y_center - LINE_Y) <= LINE_Y_TOLERANCE if label_name == 'person' else False
                                window_active = (event_highlight_until.get(entity_id, 0) >= frame_idx) if label_name == 'person' else False
                                highlight_active = in_tol_now or window_active
                                if highlight_active:
                                    frame_has_highlight = True
                                draw_tracked_box(frame, box, local_id, label_name, conf, soc, highlight=highlight_active, display_id=disp_id)
                                if label_name == 'person':
                                    accepted_persons.append((box[:], entity_id))
                                logger.debug("[%s] Person p%s re-acquired at frame %d; side changed %s -> %s (inferred crossing)", file_basename, disp_id, frame_idx, prev_global_side, current_side)
                                # Skip normal draw below for this object to avoid double drawing
                                event_last_side_global[global_id] = current_side
                                continue
                            else:
                                event_first_side[entity_id] = current_side
                                event_last_side[entity_id] = current_side
                                if label_name == 'person':
                                    logger.debug("[%s] Person p%s appeared at frame %d %s the line", file_basename, disp_id, frame_idx, current_side)
                        else:
                            prev_y = event_prev_y[entity_id]
                            visual_crossed = False
                            # Apply tolerance band to reduce jitter-based flips
                            if prev_y <= LINE_Y - LINE_Y_TOLERANCE and y_center >= LINE_Y + LINE_Y_TOLERANCE:
                                event_cross_dirs[entity_id].add('down')
                                if label_name == 'person':
                                    logger.debug("[%s] Person p%s logical crossing 'down' detected at frame %d (tolerance band)", file_basename, disp_id, frame_idx)
                            elif prev_y >= LINE_Y + LINE_Y_TOLERANCE and y_center <= LINE_Y - LINE_Y_TOLERANCE:
                                event_cross_dirs[entity_id].add('up')
                                if label_name == 'person':
                                    logger.debug("[%s] Person p%s logical crossing 'up' detected at frame %d (tolerance band)", file_basename, disp_id, frame_idx)
                            # Hysteresis-based crossing + dwell gating: use stable side outside tolerance band
                            current_stable_side = None
                            if y_center <= LINE_Y - LINE_Y_TOLERANCE:
                                current_stable_side = 'above'
                            elif y_center >= LINE_Y + LINE_Y_TOLERANCE:
                                current_stable_side = 'below'
                            # Track entry into tolerance band for explicit logging
                            in_tol = abs(y_center - LINE_Y) <= LINE_Y_TOLERANCE
                            was_in_tol = event_in_tolerance.get(entity_id, False)
                            if in_tol and not was_in_tol and label_name == 'person':
                                logger.debug("[%s] Person p%s entered line tolerance at frame %d", file_basename, disp_id, frame_idx)
                                # Start a minimum highlight window from the entry moment
                                event_highlight_until[entity_id] = max(event_highlight_until.get(entity_id, 0), frame_idx + HIGHLIGHT_WINDOW_FRAMES)
                            elif (not in_tol) and was_in_tol and label_name == 'person':
                                logger.debug("[%s] Person p%s left line tolerance at frame %d", file_basename, disp_id, frame_idx)
                            event_in_tolerance[entity_id] = in_tol
                            # Update stability counters
                            if current_stable_side is not None:
                                event_stable_frames[entity_id] = event_stable_frames.get(entity_id, 0) + 1
                            else:
                                event_stable_frames[entity_id] = 0
                            prev_stable = event_stable_side.get(entity_id)
                            if current_stable_side is not None and event_stable_frames.get(entity_id, 0) >= STABLE_MIN_FRAMES:
                                if prev_stable is None:
                                    # First time reaching stability
                                    event_stable_side[entity_id] = current_stable_side
                                    event_stable_since[entity_id] = frame_idx
                                    # Clear any stale pending
                                    if entity_id in event_pending_cross:
                                        del event_pending_cross[entity_id]
                                    if label_name == 'person':
                                        logger.debug("[%s] Person p%s stable '%s' at frame %d", file_basename, disp_id, current_stable_side, frame_idx)
                                elif current_stable_side != prev_stable:
                                    # Stable side changed → start pending crossing that must pass dwell
                                    new_dir = None
                                    if prev_stable == 'below' and current_stable_side == 'above':
                                        new_dir = 'up'
                                    elif prev_stable == 'above' and current_stable_side == 'below':
                                        new_dir = 'down'
                                    event_stable_side[entity_id] = current_stable_side
                                    event_stable_since[entity_id] = frame_idx
                                    if new_dir is not None:
                                        event_pending_cross[entity_id] = {'dir': new_dir, 'since_frame': frame_idx}
                                        if label_name == 'person':
                                            logger.debug("[%s] Person p%s stable side changed to '%s' at frame %d; pending '%s' crossing started", file_basename, disp_id, current_stable_side, frame_idx, new_dir)
                                else:
                                    # Staying on same stable side → if pending for this side, confirm after dwell time
                                    pend = event_pending_cross.get(entity_id)
                                    if pend is not None:
                                        dwell_frames = frame_idx - event_stable_since.get(entity_id, frame_idx)
                                        if dwell_frames >= int(DWELL_SECONDS * fps):
                                            event_cross_dirs[entity_id].add(pend['dir'])
                                            del event_pending_cross[entity_id]
                                            if label_name == 'person':
                                                logger.debug("[%s] Person p%s crossing '%s' confirmed after dwell (%d frames) at frame %d", file_basename, disp_id, pend['dir'], dwell_frames, frame_idx)
                            # Visual crossing detection without tolerance (for highlight only)
                            if (prev_y < LINE_Y <= y_center) or (prev_y > LINE_Y >= y_center):
                                visual_crossed = True
                            event_prev_y[entity_id] = y_center
                            event_last_side[entity_id] = current_side
                            if visual_crossed:
                                frame_has_crossing = True
                                # Person-in-ROI check for this detection (so crossing frames still count toward ROI presence)
                                cx = int((box[0] + box[2]) / 2)
                                cy = int((box[1] + box[3]) / 2)
                                if cv2.pointPolygonTest(roi_polygon_cv, (cx, cy), False) >= 0:
                                    frame_has_person_in_roi = True
                                # Draw highlight if crossing detected on this frame
                                # Non-person labels don't use disp_id
                                if label_name != 'person':
                                    disp_id = None
                                # Highlight if in tolerance or within minimum highlight window
                                in_tol_now = abs(y_center - LINE_Y) <= LINE_Y_TOLERANCE if label_name == 'person' else False
                                window_active = (event_highlight_until.get(entity_id, 0) >= frame_idx) if label_name == 'person' else False
                                highlight_active = in_tol_now or window_active
                                if highlight_active:
                                    frame_has_highlight = True
                                draw_tracked_box(frame, box, local_id, label_name, conf, soc, highlight=highlight_active, display_id=disp_id)
                                if label_name == 'person':
                                    logger.debug("[%s] Person p%s visually crossed the line at frame %d (prev_y=%d -> y=%d)", file_basename, disp_id, frame_idx, prev_y, y_center)
                                    accepted_persons.append((box[:], entity_id))
                                # Skip normal draw below for this object to avoid double drawing
                                continue

                        # Update last known side for this tracker id (used to bridge gaps)
                        event_last_side_global[global_id] = current_side

                    # Person-in-ROI check using center point inside ROI polygon
                    if label_name == 'person':
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        if cv2.pointPolygonTest(roi_polygon_cv, (cx, cy), False) >= 0:
                            frame_has_person_in_roi = True

                    # Highlight when in tolerance OR still within minimum highlight window
                    highlight_active = False
                    if label_name == 'person':
                        in_tol_now = abs(y_center - LINE_Y) <= LINE_Y_TOLERANCE
                        eid = event_global_to_entity.get(global_id)
                        window_active = (event_highlight_until.get(eid, 0) >= frame_idx) if eid is not None else False
                        highlight_active = in_tol_now or window_active
                        if highlight_active:
                            frame_has_highlight = True
                    # Determine display id for person labels (already set earlier)
                    disp_id = None if label_name != 'person' else disp_id
                    draw_tracked_box(frame, box, local_id, label_name, conf, soc, highlight=highlight_active, display_id=disp_id)
                    if label_name == 'person' and disp_id is not None:
                        accepted_persons.append((box[:], event_global_to_entity.get(global_id, entity_id if 'entity_id' in locals() else -1)))

            if frame_has_person_in_roi:
                person_frames_in_roi += 1

            cv2.line(frame, (0, LINE_Y), (orig_w, LINE_Y), COLOR_LINE, OVERLAY_LINE_THICKNESS)
            cv2.putText(frame, f"Hvirtka Y={LINE_Y}", (10, LINE_Y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, OVERLAY_FONT_SCALE, COLOR_LINE, OVERLAY_TEXT_THICKNESS, cv2.LINE_AA)

            # Draw static event overlay in bottom-right: "<idx>/<total> - MM:SS"
            current_seconds = int((frame_idx - 1) / fps)
            draw_event_overlay(frame, clip_index + 1, len(significant_sub_clips), current_seconds)

            # Append frames to output based on output_stride to speed up render without
            # sacrificing tracking continuity (for moderate-length events)
            append_current_frame = (processed_offset % output_stride == 0) or frame_has_crossing or frame_has_highlight
            if append_current_frame:
                if prof.enabled:
                    t0 = time.perf_counter()
                # Store frames already in output resolution to avoid keeping full 4K frames in RAM.
                # This reduces memory pressure substantially on low-power machines.
                h_out, w_out = output_size[1], output_size[0]
                frame_out = frame
                if frame.shape[0] != h_out or frame.shape[1] != w_out:
                    render_resize_interpolation = cv2.INTER_LINEAR if is_4k else cv2.INTER_AREA
                    frame_out = cv2.resize(frame, (w_out, h_out), interpolation=render_resize_interpolation)
                rgb_frame = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
                event_frames_rgb.append(rgb_frame)
                if prof.enabled:
                    prof.add("track.render_append", time.perf_counter() - t0)
                    prof.counts["track.frames_appended"] += 1

            # Collect ReID candidate crops near line tolerance every REID_SAMPLING_STRIDE
            # Use clean frame copy to avoid drawn overlays in crops
            if needs_reid and frame_for_reid is not None:
                if prof.enabled:
                    t0 = time.perf_counter()
                if len(reid_candidate_crops) < REID_MAX_SAMPLES and accepted_persons:
                    for pbox, _eid in accepted_persons:
                        y_center = int((pbox[1] + pbox[3]) / 2)
                        if abs(y_center - LINE_Y) <= (LINE_Y_TOLERANCE + REID_LINE_EXTRA_TOLERANCE):
                            x1 = max(0, int(pbox[0]) - REID_CROP_PADDING)
                            y1 = max(0, int(pbox[1]) - REID_CROP_PADDING)
                            x2 = min(orig_w, int(pbox[2]) + REID_CROP_PADDING)
                            y2 = min(orig_h, int(pbox[3]) + REID_CROP_PADDING)
                            if x2 > x1 and y2 > y1:
                                crop = frame_for_reid[y1:y2, x1:x2]
                                if crop is not None and crop.size > 0:
                                    reid_candidate_crops.append(crop)
                if prof.enabled:
                    prof.add("track.reid_sampling", time.perf_counter() - t0)

        # Confirm any last pending crossing if it matches the final side
        for pid, pend in list(event_pending_cross.items()):
            end_side = event_last_side.get(pid, event_first_side.get(pid))
            target_side = 'above' if pend['dir'] == 'up' else 'below'
            if end_side == target_side:
                if pid not in event_cross_dirs:
                    event_cross_dirs[pid] = set()
                event_cross_dirs[pid].add(pend['dir'])

        # Reduce per-person crossings to net counts (final-side first, otherwise both directions)
        # - If final side != start side: count 1 in that direction only
        # - If final side == start side: count 1 up and 1 down only if both directions occurred
        event_up_final = 0
        event_down_final = 0
        for pid in event_prev_y.keys():
            start_side = event_first_side.get(pid)
            end_side = event_last_side.get(pid, start_side)
            dirs = event_cross_dirs.get(pid, set())
            if start_side and end_side and start_side != end_side:
                if start_side == 'below' and end_side == 'above':
                    event_up_final += 1
                elif start_side == 'above' and end_side == 'below':
                    event_down_final += 1
            else:
                if 'up' in dirs and 'down' in dirs:
                    event_up_final += 1
                    event_down_final += 1

        event_persons_up = event_up_final
        event_persons_down = event_down_final

        # Decide whether to include this event based on person frames within ROI
        if person_frames_in_roi >= PERSON_MIN_FRAMES:
            logger.info(f"[{file_basename}] {clip_index + 1} - Event at {(start_frame / fps):.1f}s kept: person present in ROI for {person_frames_in_roi} frames (>= {PERSON_MIN_FRAMES}).")
            # Initialize CRF-based H.264 writer lazily on first accepted event
            if writer is None:
                logger.info(f"[{file_basename}] Initializing CRF-based H.264 writer (libx264, preset=faster, crf=28) at {output_size[0]}x{output_size[1]}...")
                writer = FFMPEG_VideoWriter(
                    output_filename,
                    size=output_size,
                    fps=fps,
                    codec='libx264',
                    preset='faster',
                    threads=0,
                    ffmpeg_params=['-crf', '28', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
                )
            if prof.enabled:
                t0 = time.perf_counter()
            for rgb_frame in event_frames_rgb:
                # Frames are stored in output size; keep a safety check.
                h_out, w_out = output_size[1], output_size[0]
                if rgb_frame.shape[0] != h_out or rgb_frame.shape[1] != w_out:
                    rgb_frame = cv2.resize(rgb_frame, (w_out, h_out), interpolation=cv2.INTER_AREA)
                writer.write_frame(rgb_frame)
            if prof.enabled:
                prof.add("encode.write_frames", time.perf_counter() - t0, count=len(event_frames_rgb))
            written_frame_count += len(event_frames_rgb)
            # Seed session-wide display IDs from the first kept event
            if not session_display_initialized:
                if event_display_ids:
                    for lid, did in event_display_ids.items():
                        person_display_ids[lid] = did
                    person_display_id_counter = max(event_display_ids.values()) + 1
                session_display_initialized = True
            # Merge event stats into overall
            unique_objects_detected['person'].update(event_unique_objects['person'])
            unique_objects_detected['car'].update(event_unique_objects['car'])
            persons_up += event_persons_up
            persons_down += event_persons_down
        else:
            logger.info(f"[{file_basename}] {clip_index + 1} - Event at {(start_frame / fps):.1f}s discarded: person in ROI only {person_frames_in_roi} frames (< {PERSON_MIN_FRAMES}).")
            # Save a representative middle frame for significant-but-no-person events
            if SAVE_INSIGNIFICANT_FRAMES:
                try:
                    mid_frame_index = start_frame + (end_frame - start_frame) // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame_path = detect_draw_and_save_snapshot(
                            frame=frame,
                            soc=soc,
                            output_dir=output_dir,
                            input_video_path=input_video_path,
                            file_basename=file_basename,
                            mid_frame_index=mid_frame_index,
                            tag="no_person",
                            classes=DETECT_CLASSES,
                        )
                        if frame_path and SEND_INSIGNIFICANT_FRAMES:
                            insignificant_motion_frames.append(frame_path)
                except Exception as e:
                    logger.warning(f"[{file_basename}] Failed to save no_person frame: {e}")

    cap.release()

    if written_frame_count == 0:
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] No significant events with person in ROI found. Full processing took {elapsed_time:.2f} seconds.")
        _maybe_log_profile("no_person")
        return {'status': 'no_person', 'clip_path': None, 'insignificant_frames': insignificant_motion_frames}

    # Finalize CRF-based H.264 writer
    logger.debug("[%s] Finalizing highlight clip...", file_basename)
    if writer is not None:
        writer.close()

    file_duration = written_frame_count / fps
    file_size = os.path.getsize(output_filename) / (1024 * 1024)
    logger.info(f"[{file_basename}] Successfully created clip: {output_filename}. Duration: {file_duration:.2f}s, Size: {file_size:.2f} MB")

    num_persons = len(unique_objects_detected['person'])
    num_cars = len(unique_objects_detected['car'])
    crossing_detected = persons_up > 0 or persons_down > 0

    final_status = 'significant_motion'
    crossing_direction = None
    if crossing_detected:
        final_status = 'gate_crossing'
        if persons_up > 0 and persons_down > 0:
            crossing_direction = 'both'
        elif persons_up > 0:
            crossing_direction = 'up'
        else:
            crossing_direction = 'down'
        logger.info(f"[{file_basename}] Gate crossing detected! Direction: {crossing_direction}. Persons Up: {persons_up}, Down: {persons_down}.")

    # Person ReID run: evaluate collected crops against gallery
    reid_result = None
    if REID_ENABLED:
        reid_result = {"matched": False, "score": 0.0, "threshold": REID_THRESHOLD, "samples": 0, "best_path": None}
        try:
            samples = reid_candidate_crops
            reid_result["samples"] = len(samples)
            if len(samples) > 0:
                logger.info(f"[{file_basename}] Running person ReID on {len(samples)} crop(s) with stride={REID_SAMPLING_STRIDE}.")
                reid = PersonReID(
                    REID_MODEL_PATH,
                    REID_GALLERY_PATH,
                    threshold=REID_THRESHOLD,
                    file_basename=file_basename,
                    negative_gallery_path=REID_NEGATIVE_GALLERY_PATH,
                    negative_margin=REID_NEGATIVE_MARGIN,
                )

                # Score each sample vs gallery and compute embeddings for diversity filtering
                scored = []
                best_score = 0.0
                for crop in samples:
                    try:
                        emb = reid.get_embedding(crop)
                        score = 0.0
                        neg_score = 0.0
                        if len(reid.gallery_vectors) > 0:
                            # compute max cosine similarity to gallery
                            for ref_vec in reid.gallery_vectors:
                                s = float(np.dot(emb, ref_vec))
                                if s > score:
                                    score = s
                        if len(reid.negative_vectors) > 0:
                            for neg_vec in reid.negative_vectors:
                                s = float(np.dot(emb, neg_vec))
                                if s > neg_score:
                                    neg_score = s
                        scored.append({"crop": crop, "score": float(score), "neg": float(neg_score), "emb": emb})
                        if score > best_score:
                            best_score = score
                    except Exception:
                        continue

                # Margin-based decision vs negatives (if present)
                best_neg = 0.0
                if scored:
                    try:
                        best_neg = max(item.get("neg", 0.0) for item in scored)
                    except Exception:
                        best_neg = 0.0
                reid_result["matched"] = (best_score >= REID_THRESHOLD) and ((best_score - best_neg) >= REID_NEGATIVE_MARGIN)
                reid_result["score"] = round(best_score, 4)
                reid_result["neg_score"] = round(best_neg, 4)
                reid_result["margin"] = REID_NEGATIVE_MARGIN

                # Select up to REID_TOP_K diverse top crops
                selected = []
                if scored:
                    scored.sort(key=lambda x: x["score"], reverse=True)
                    for cand in scored:
                        if len(selected) >= max(1, int(REID_TOP_K)):
                            break
                        ok = True
                        for s in selected:
                            # cosine distance between normalized embeddings
                            try:
                                sim = float(np.dot(cand["emb"], s["emb"]))
                            except Exception:
                                sim = 1.0
                            if (1.0 - sim) < REID_DIVERSITY_MIN_DIST:
                                ok = False
                                break
                        if ok:
                            selected.append(cand)
                    # If couldn't reach K due to diversity, backfill from remaining best
                    if len(selected) < max(1, int(REID_TOP_K)):
                        for cand in scored:
                            # Avoid Numpy array equality on dicts; use identity
                            if any(cand is s for s in selected):
                                continue
                            selected.append(cand)
                            if len(selected) >= max(1, int(REID_TOP_K)):
                                break

                # Save selected crops (indexed only, no legacy duplicate)
                if SAVE_REID_BEST_CROP and selected:
                    try:
                        date_folder = datetime.now().strftime("%Y%m%d")
                        daily_dir = os.path.join(output_dir, date_folder)
                        os.makedirs(daily_dir, exist_ok=True)
                        dt = parse_datetime_from_path(input_video_path)
                        hour = dt.strftime('%H') if dt else (input_video_path.split(os.path.sep)[-2][-2:] if len(input_video_path.split(os.path.sep)) >= 2 else "")

                        saved_paths = []
                        for idx, item in enumerate(selected, start=1):
                            fname_idxed = f"{hour}H{os.path.splitext(file_basename)[0]}_reid_best{idx}.jpg"
                            save_path_idxed = os.path.join(daily_dir, fname_idxed)
                            cv2.imwrite(save_path_idxed, item["crop"], [cv2.IMWRITE_JPEG_QUALITY, 90])
                            saved_paths.append(save_path_idxed)
                        # best_path points to the highest-score indexed file
                        reid_result["best_path"] = saved_paths[0] if saved_paths else None
                        logger.info(f"[{file_basename}] ReID topK saved {len(selected)} crops.")
                    except Exception as e:
                        logger.warning(f"[{file_basename}] Failed to save ReID crops: {e}")

                logger.info(f"[{file_basename}] ReID result: matched={reid_result['matched']}, pos={best_score:.3f}, neg={best_neg:.3f}, delta={(best_score - best_neg):.3f}, thr={REID_THRESHOLD:.3f}, margin={REID_NEGATIVE_MARGIN:.3f}.")
            else:
                logger.info(f"[{file_basename}] No ReID candidate crops collected.")
        except Exception as e:
            logger.warning(f"[{file_basename}] ReID evaluation failed: {e}")
    else:
        logger.debug("[%s] ReID is disabled (REID_ENABLED=False). Skipping identification.", file_basename)

    elapsed_time = time.time() - start_time
    logger.info(f"[{file_basename}] Full processing took {elapsed_time:.2f} seconds. Detected: {num_persons} persons, {num_cars} cars.")

    _maybe_log_profile(final_status)

    return {
        'status': final_status,
        'clip_path': output_filename,
        'insignificant_frames': insignificant_motion_frames,
        'persons_detected': num_persons,
        'cars_detected': num_cars,
        'crossing_direction': crossing_direction,
        'persons_up': persons_up,
        'persons_down': persons_down,
        'reid': reid_result,
    }
