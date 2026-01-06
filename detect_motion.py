import os
import time
import json
import logging

import cv2
import numpy as np
from datetime import datetime

from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Motion detection configuration ---
MIN_CONTOUR_AREA = 1800
ROI_CONFIG_FILE = os.path.join(SCRIPT_DIR, "roi.json")
PADDING_SECONDS = 1.5
WARMUP_FRAMES = 15
MAX_EVENT_GAP_SECONDS = 3.0
MIN_EVENT_DURATION_SECONDS = 0.8
MIN_INSIGNIFICANT_EVENT_DURATION_SECONDS = 0.2
# Tracking/render speed-up thresholds
# <= TRACK_FULL_UNTIL_SECONDS: track all frames, render all frames
# > TRACK_FULL_UNTIL_SECONDS and <= TRACK_SKIP_FROM_SECONDS: track all frames, render every 2nd frame
# > TRACK_SKIP_FROM_SECONDS: track every 2nd frame, render every 2nd frame (legacy behavior)
TRACK_FULL_UNTIL_SECONDS = 6.0
TRACK_SKIP_FROM_SECONDS = 12.0
SAVE_INSIGNIFICANT_FRAMES = True
SEND_INSIGNIFICANT_FRAMES = False
CROP_PADDING = 30
TRACK_ROI_PADDING = 10  # padding for tracker ROI bounding box
PERSON_MIN_FRAMES = 10

# --- Tesla config ---
TESLA_EMAIL = os.getenv("TESLA_EMAIL")
TESLA_REFRESH_TOKEN = os.getenv("TESLA_REFRESH_TOKEN")
TESLA_SOC_FILE = os.path.join(SCRIPT_DIR, "temp", "tesla_soc.txt")
TESLA_LAST_CHECK = 0
TESLA_SOC_CHECK_ENABLED = bool(teslapy and TESLA_REFRESH_TOKEN and TESLA_EMAIL)

# --- Object Detection Configuration ---
OBJECT_DETECTION_MODEL_PATH = os.getenv("OBJECT_DETECTION_MODEL_PATH", default="best_openvino_model")
CONF_THRESHOLD = 0.5
DETECT_CLASSES = [0, 1]  # 0: person, 1: car
TRACK_ROI_ENABLED = True  # Enable tracker ROI crop (from roi.json: 'tracker_roi' or fallback to motion_detection_roi/legacy)
LINE_Y = 860
COLOR_PERSON = (100, 200, 0)
COLOR_CAR = (200, 120, 0)
COLOR_DEFAULT = (255, 255, 255)
COLOR_HIGHLIGHT = (80, 90, 245)
COLOR_LINE = (0, 255, 255)
LINE_Y_TOLERANCE = 6
HIGHLIGHT_WINDOW_FRAMES = 5 # Minimum highlight duration (frames) after entering tolerance band
STABLE_MIN_FRAMES = 2  # frames outside tolerance required to confirm stable side
DWELL_SECONDS = 2.0    # seconds to stay on the other side to confirm a crossing

# --- Load Object Detection Model ---
try:
    object_detection_model = YOLO(OBJECT_DETECTION_MODEL_PATH, task='detect')
    logger.info(f"Object detection model loaded successfully from {OBJECT_DETECTION_MODEL_PATH}.")
except Exception as e:
    logger.critical(f"Failed to load object detection model: {e}", exc_info=True)
    raise


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
        tx, ty = 1150, 450
        if label_name == 'car':
            if x1 <= tx <= x2 and y1 <= ty <= y2:
                if soc is not None:
                    label_text = f"Tesla {conf:.0%} / SoC {soc}%"
                else:
                    label_text = f"Tesla {conf:.0%}"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)

    cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)
    cv2.putText(frame, label_text, (x1, y1 - 5),
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)


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
    font_scale = 1
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    pad_x = 20
    pad_y = 14

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

        # Draw line overlay similar to main path
        cv2.line(frame, (0, LINE_Y), (frame.shape[1], LINE_Y), COLOR_LINE, 1)
        cv2.putText(frame, f"Hvirtka Y={LINE_Y}", (10, LINE_Y - 10),
                    cv2.FONT_HERSHEY_DUPLEX, 1, COLOR_LINE, 1, cv2.LINE_AA)

        # Save to daily folder inside output dir (YYYYMMDD)
        date_folder = datetime.now().strftime("%Y%m%d")
        daily_dir = os.path.join(output_dir, date_folder)
        os.makedirs(daily_dir, exist_ok=True)
        cam_prefix = input_video_path.split(os.path.sep)[-2][-2:] if len(input_video_path.split(os.path.sep)) >= 2 else ""
        frame_filename = f"{cam_prefix}H{os.path.splitext(file_basename)[0]}_{tag}_{mid_frame_index}.jpg"
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
    # Read roi.json once and derive required polygons (motion, tracker, person)
    roi_config = read_roi_config(ROI_CONFIG_FILE)
    if roi_config is None:
        logger.error(f"[{file_basename}] ROI config file not found or invalid: {ROI_CONFIG_FILE}.")
        return {'status': 'error', 'clip_path': None, 'insignificant_frames': []}

    # Motion detection ROI: if dict, expect 'motion_detection_roi'; if legacy (list), use it directly
    if isinstance(roi_config, dict):
        roi_poly_points = as_np_points(roi_config.get('motion_detection_roi'))
    elif isinstance(roi_config, list):
        roi_poly_points = as_np_points(roi_config)
    else:
        roi_poly_points = None
    if roi_poly_points is None:
        logger.error(f"[{file_basename}] Motion detection ROI missing in roi.json.")
        return {'status': 'error', 'clip_path': None, 'insignificant_frames': []}

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.error(f"[{file_basename}] Could not open video file {input_video_path}")
        return {'status': 'error', 'clip_path': None, 'insignificant_frames': []}

    start_time = time.time()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
            person_tracker_polygon_cv = person_tracker_roi_points.reshape((-1, 1, 2))
        except Exception:
            person_tracker_polygon_cv = None

    logger.info(
        f"[{file_basename}] Original frame: {orig_w}x{orig_h}. motion_detection_roi={crop_w}x{crop_h}, "
        f"tracker_roi={(f'{track_roi_bbox[2]-track_roi_bbox[0]}x{track_roi_bbox[3]-track_roi_bbox[1]}' if track_roi_bbox is not None else 'False')}, "
        f"person_tracker_roi={(f'{int(np.max(person_tracker_roi_points[:,0]) - np.min(person_tracker_roi_points[:,0]))}x{int(np.max(person_tracker_roi_points[:,1]) - np.min(person_tracker_roi_points[:,1]))}' if person_tracker_polygon_cv is not None else 'False')}"
    )

    backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=600, detectShadows=True)
    roi_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [analysis_roi_points], 255)

    logger.debug("[%s] Starting smart background model pre-training...", file_basename)
    pre_trained = False
    training_candidate_times = [30, 40, 50, 20]

    frames_to_train = 5 * int(fps)
    # Validate candidate segment across the entire training window to avoid
    # pretraining on periods with subtle motion.
    frames_to_sample = frames_to_train

    for start_sec in training_candidate_times:
        start_frame = int(start_sec * fps)
        if total_frames < start_frame + frames_to_sample:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        motion_detected_in_segment = False
        temp_backSub = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=400, detectShadows=True)

        for i in range(frames_to_sample):
            ret, frame = cap.read()
            if not ret:
                break

            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
            fg_mask = temp_backSub.apply(roi_frame)

            if i > 5 and cv2.countNonZero(fg_mask) > (roi_mask.size * 0.15):
                motion_detected_in_segment = True
                logger.info(f"[{file_basename}] Motion detected in pre-training candidate segment at {start_sec}s. Trying next segment.")
                break

        if not motion_detected_in_segment:
            logger.info(f"[{file_basename}] Found a static segment at {start_sec}s. Pre-training background model...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(frames_to_train):
                ret, frame = cap.read()
                if not ret:
                    break

                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
                backSub.apply(roi_frame)

            pre_trained = True
            break

    if pre_trained:
        logger.info(f"[{file_basename}] Background model pre-trained. Resetting to start for analysis.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        logger.warning(f"[{file_basename}] Could not find a static segment to pre-train model. Using standard warm-up.")

    EFFECTIVE_WARMUP = 5 if pre_trained else WARMUP_FRAMES

    motion_events = []
    for frame_index in range(EFFECTIVE_WARMUP, total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
        fg_mask = backSub.apply(roi_frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_enough_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if large_enough_contours:
            motion_events.append((frame_index, large_enough_contours))

    if not motion_events:
        logger.info(f"[{file_basename}] No significant motion found.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
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
        return {'status': 'no_motion', 'clip_path': None, 'insignificant_frames': []}

    if not significant_sub_clips:
        logger.info(f"[{file_basename}] No significant long-duration motion found.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        return {'status': 'no_significant_motion', 'clip_path': None, 'insignificant_frames': insignificant_motion_frames}

    logger.info(f"[{file_basename}] Starting object tracking for {len(significant_sub_clips)} significant event(s).")
    output_filename = os.path.join(output_dir, file_basename)

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

    # Process each event separately and include only those with person inside ROI for >= PERSON_MIN_FRAMES
    for clip_index, (start_frame, end_frame) in enumerate(significant_sub_clips):
        duration_seconds = (end_frame - start_frame) / fps
        is_long_motion = duration_seconds > TRACK_FULL_UNTIL_SECONDS
        padding_seconds_adjusted = 1 if is_long_motion else PADDING_SECONDS
        padded_start = max(0, start_frame - int(padding_seconds_adjusted * fps))
        padded_end = min(total_frames, end_frame + int(padding_seconds_adjusted * fps))

        if clip_index == 0 and start_frame <= (EFFECTIVE_WARMUP + fps * 1.0):
            logger.info(f"[{file_basename}] Motion starts at the beginning. Including video from frame 0.")
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
            logger.info(f"[{file_basename}] Long motion event ({duration_seconds:.2f}s). tracker_stride={tracker_stride}, output_stride={output_stride}.")

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
        event_display_id_counter = 1
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
        person_frames_in_roi = 0

        # Seek once to the start of the event, then advance sequentially to avoid decoder seek artifacts
        cap.set(cv2.CAP_PROP_POS_FRAMES, padded_start)
        frame_idx = padded_start
        while frame_idx < padded_end:
            grabbed = cap.grab()
            if not grabbed:
                frame_idx += 1
                continue

            # Skip frames for tracking based on tracker_stride
            if (frame_idx - padded_start) % tracker_stride != 0:
                frame_idx += 1
                continue

            ret, frame = cap.retrieve()
            frame_idx += 1
            if not ret or frame is None:
                continue

            # Apply tracker ROI crop if available
            if track_roi_bbox is not None:
                tx1, ty1, tx2, ty2 = track_roi_bbox
                track_frame = frame[ty1:ty2, tx1:tx2]
            else:
                track_frame = frame

            results = object_detection_model.track(
                track_frame, imgsz=640, conf=CONF_THRESHOLD, persist=True, verbose=False,
                tracker="tracker.yaml", classes=DETECT_CLASSES
            )

            # Count whether this frame contains a person within ROI and whether any crossing/highlight happens
            frame_has_person_in_roi = False
            frame_has_crossing = False
            frame_has_highlight = False

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
                # Prevent multiple detections mapping to same entity within a single frame
                frame_assigned_entities = set()

                for box, global_id, cls, conf in zip(boxes, global_ids, clss, confs):
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

                        # Determine display id: event-local until first kept event, then session-wide
                        disp_id = None
                        if label_name == 'person':
                            if session_display_initialized:
                                if local_id not in person_display_ids:
                                    person_display_ids[local_id] = person_display_id_counter
                                    person_display_id_counter += 1
                                disp_id = person_display_ids[local_id]
                            else:
                                if local_id not in event_display_ids:
                                    event_display_ids[local_id] = event_display_id_counter
                                    event_display_id_counter += 1
                                disp_id = event_display_ids[local_id]

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
                                # Determine display id for label
                                if label_name == 'person':
                                    if session_display_initialized:
                                        if local_id not in person_display_ids:
                                            person_display_ids[local_id] = person_display_id_counter
                                            person_display_id_counter += 1
                                        disp_id = person_display_ids[local_id]
                                    else:
                                        if local_id not in event_display_ids:
                                            event_display_ids[local_id] = event_display_id_counter
                                            event_display_id_counter += 1
                                        disp_id = event_display_ids[local_id]
                                else:
                                    disp_id = None
                                # Highlight if currently in tolerance or still within minimum highlight window
                                in_tol_now = abs(y_center - LINE_Y) <= LINE_Y_TOLERANCE if label_name == 'person' else False
                                window_active = (event_highlight_until.get(entity_id, 0) >= frame_idx) if label_name == 'person' else False
                                highlight_active = in_tol_now or window_active
                                if highlight_active:
                                    frame_has_highlight = True
                                draw_tracked_box(frame, box, local_id, label_name, conf, soc, highlight=highlight_active, display_id=disp_id)
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
                                # Determine display id for label
                                if label_name == 'person':
                                    if session_display_initialized:
                                        if local_id not in person_display_ids:
                                            person_display_ids[local_id] = person_display_id_counter
                                            person_display_id_counter += 1
                                        disp_id = person_display_ids[local_id]
                                    else:
                                        if local_id not in event_display_ids:
                                            event_display_ids[local_id] = event_display_id_counter
                                            event_display_id_counter += 1
                                        disp_id = event_display_ids[local_id]
                                else:
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
                    # Determine display id for person labels
                    disp_id = None
                    if label_name == 'person':
                        if session_display_initialized:
                            if local_id not in person_display_ids:
                                person_display_ids[local_id] = person_display_id_counter
                                person_display_id_counter += 1
                            disp_id = person_display_ids[local_id]
                        else:
                            if local_id not in event_display_ids:
                                event_display_ids[local_id] = event_display_id_counter
                                event_display_id_counter += 1
                            disp_id = event_display_ids[local_id]
                    draw_tracked_box(frame, box, local_id, label_name, conf, soc, highlight=highlight_active, display_id=disp_id)

            if frame_has_person_in_roi:
                person_frames_in_roi += 1

            cv2.line(frame, (0, LINE_Y), (orig_w, LINE_Y), COLOR_LINE, 1)
            cv2.putText(frame, f"Hvirtka Y={LINE_Y}", (10, LINE_Y - 10),
                        cv2.FONT_HERSHEY_DUPLEX, 1, COLOR_LINE, 1, cv2.LINE_AA)

            # Draw static event overlay in bottom-right: "<idx>/<total> - MM:SS"
            current_seconds = int((frame_idx - 1) / fps)
            draw_event_overlay(frame, clip_index + 1, len(significant_sub_clips), current_seconds)

            # Append frames to output based on output_stride to speed up render without
            # sacrificing tracking continuity (for moderate-length events)
            processed_offset = (frame_idx - 1) - padded_start
            append_current_frame = (processed_offset % output_stride == 0) or frame_has_crossing or frame_has_highlight
            if append_current_frame:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                event_frames_rgb.append(rgb_frame)

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
            logger.info(f"[{file_basename}] Event at {(start_frame / fps):.1f}s kept: person present in ROI for {person_frames_in_roi} frames (>= {PERSON_MIN_FRAMES}).")
            # Initialize CRF-based H.264 writer lazily on first accepted event
            if writer is None:
                logger.info(f"[{file_basename}] Initializing CRF-based H.264 writer (libx264, preset=medium, crf=28)...")
                writer = FFMPEG_VideoWriter(
                    output_filename,
                    size=(orig_w, orig_h),
                    fps=fps,
                    codec='libx264',
                    preset='medium',
                    threads=2,
                    ffmpeg_params=['-crf', '28', '-pix_fmt', 'yuv420p', '-movflags', '+faststart']
                )
            for rgb_frame in event_frames_rgb:
                writer.write_frame(rgb_frame)
            written_frame_count += len(event_frames_rgb)
            # Seed session-wide display IDs from the first kept event
            if not session_display_initialized:
                if event_display_ids:
                    # Preserve numbering used in this event
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
            logger.info(f"[{file_basename}] Event at {(start_frame / fps):.1f}s discarded: person in ROI only {person_frames_in_roi} frames (< {PERSON_MIN_FRAMES}).")
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

    elapsed_time = time.time() - start_time
    logger.info(f"[{file_basename}] Full processing took {elapsed_time:.2f} seconds. Detected: {num_persons} persons, {num_cars} cars.")

    return {
        'status': final_status,
        'clip_path': output_filename,
        'insignificant_frames': insignificant_motion_frames,
        'persons_detected': num_persons,
        'cars_detected': num_cars,
        'crossing_direction': crossing_direction,
        'persons_up': persons_up,
        'persons_down': persons_down
    }
