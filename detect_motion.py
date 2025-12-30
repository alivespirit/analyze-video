import os
import time
import json
import logging

import cv2
import numpy as np
from datetime import datetime

from moviepy import ImageSequenceClip

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
SEND_INSIGNIFICANT_FRAMES = True
CROP_PADDING = 30
MAX_BOX_AREA_PERCENT = 0.80
PERSON_MIN_FRAMES = 8

# --- Tesla config ---
TESLA_EMAIL = os.getenv("TESLA_EMAIL")
TESLA_REFRESH_TOKEN = os.getenv("TESLA_REFRESH_TOKEN")
TESLA_SOC_FILE = os.path.join(SCRIPT_DIR, "temp", "tesla_soc.txt")
TESLA_LAST_CHECK = 0
TESLA_SOC_CHECK_ENABLED = bool(teslapy and TESLA_REFRESH_TOKEN and TESLA_EMAIL)

# --- Object Detection Configuration ---
OBJECT_DETECTION_MODEL_PATH = os.getenv("OBJECT_DETECTION_MODEL_PATH", default="best_openvino_model")
CONF_THRESHOLD = 0.5
LINE_Y = 860
LINE_CROSSING_COOLDOWN_SECONDS = 3.0
COLOR_PERSON = (100, 200, 0)
COLOR_CAR = (200, 120, 0)
COLOR_DEFAULT = (255, 255, 255)
COLOR_LINE = (0, 255, 255)

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


def load_roi(file_path):
    """
    Loads the Region of Interest (ROI) polygon points from a JSON file.

    Args:
        file_path (str): The path to the roi.json file.

    Returns:
        np.ndarray: A NumPy array of points defining the ROI, or None if the file doesn't exist.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            points = json.load(f)
        return np.array(points, dtype=np.int32)
    return None


def draw_tracked_box(frame, box, local_id, label_name, conf, soc):
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

    if label_name == 'person':
        color = COLOR_PERSON
    elif label_name == 'car':
        color = COLOR_CAR
    else:
        color = COLOR_DEFAULT

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
    roi_poly_points = load_roi(ROI_CONFIG_FILE)
    if roi_poly_points is None:
        logger.error(f"[{file_basename}] ROI config file not found.")
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

    logger.info(f"[{file_basename}] Original frame: {orig_w}x{orig_h}. Analyzing cropped region: {crop_w}x{crop_h} at ({crop_x1},{crop_y1}).")

    local_roi_points = roi_poly_points.copy()
    local_roi_points[:, 0] -= crop_x1
    local_roi_points[:, 1] -= crop_y1

    analysis_roi_points = local_roi_points.astype(np.int32)

    backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=600, detectShadows=True)
    roi_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [analysis_roi_points], 255)

    logger.debug(f"[{file_basename}] Starting smart background model pre-training...")
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

            if i > 5 and cv2.countNonZero(fg_mask) > (roi_mask.size * 0.2):
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
            logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is SIGNIFICANT. Will process with tracker.")
            significant_sub_clips.append((start_frame, end_frame))
        elif duration_seconds >= MIN_INSIGNIFICANT_EVENT_DURATION_SECONDS:
            all_shorter_than_insignificant = False
            if SEND_INSIGNIFICANT_FRAMES:
                logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is insignificant. Extracting frame.")
                mid_frame_index = start_frame + (end_frame - start_frame) // 2

                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
                ret, frame = cap.read()
                if ret:
                    results = object_detection_model.track(frame, imgsz=640, conf=CONF_THRESHOLD, persist=False, verbose=False)
                    if results[0].boxes.id is not None:
                        boxes = results[0].boxes.xyxy.cpu().tolist()
                        clss = results[0].boxes.cls.int().cpu().tolist()
                        confs = results[0].boxes.conf.float().cpu().tolist()
                        class_names = object_detection_model.names

                        local_id_counter = 1
                        for box, cls, conf in zip(boxes, clss, confs):
                            label_name = class_names[cls]
                            draw_tracked_box(frame, box, local_id_counter, label_name, conf, soc)
                            local_id_counter += 1

                    # Save to daily folder inside output dir (YYYYMMDD)
                    date_folder = datetime.now().strftime("%Y%m%d")
                    daily_dir = os.path.join(output_dir, date_folder)
                    os.makedirs(daily_dir, exist_ok=True)
                    frame_filename = f"{input_video_path.split(os.path.sep)[-2][-2:]}H{os.path.splitext(file_basename)[0]}_insignificant_{mid_frame_index}.jpg"
                    frame_path = os.path.join(daily_dir, frame_filename)
                    cv2.line(frame, (0, LINE_Y), (orig_w, LINE_Y), COLOR_LINE, 1)
                    cv2.putText(frame, f"Hvirtka Y={LINE_Y}", (10, LINE_Y - 10),
                            cv2.FONT_HERSHEY_DUPLEX, 1, COLOR_LINE, 1, cv2.LINE_AA)
                    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    #insignificant_motion_frames.append(frame_path) # disabled to keep frames for further analysis only
                    logger.info(f"[{file_basename}] Saved insignificant motion frame to {frame_path}")
            else:
                logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is insignificant. Skipping frame extraction.")
        else:
            logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is too short. Discarding as noise/shadow.")

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

    if hasattr(object_detection_model, 'predictor') and object_detection_model.predictor is not None:
        if hasattr(object_detection_model.predictor, 'trackers') and object_detection_model.predictor.trackers:
            object_detection_model.predictor.trackers[0].reset()
            logger.debug(f"[{file_basename}] Object tracker state reset.")

    # Prepare ROI polygon for point-in-polygon checks
    roi_polygon_cv = roi_poly_points.reshape((-1, 1, 2))

    all_clip_frames_rgb = []
    class_names = object_detection_model.names
    id_mapping = {}
    class_counters = {}
    unique_objects_detected = {'person': set(), 'car': set()}
    persons_up = 0
    persons_down = 0

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

            results = object_detection_model.track(frame, imgsz=640, conf=CONF_THRESHOLD, persist=True, verbose=False, tracker="tracker.yaml")

            # Count whether this frame contains a person within ROI
            frame_has_person_in_roi = False

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().tolist()
                global_ids = results[0].boxes.id.int().cpu().tolist()
                clss = results[0].boxes.cls.int().cpu().tolist()
                confs = results[0].boxes.conf.float().cpu().tolist()

                for box, global_id, cls, conf in zip(boxes, global_ids, clss, confs):
                    label_name = class_names[cls]

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
                        y_center = int((box[1] + box[3]) / 2)
                        current_side = 'above' if y_center < LINE_Y else 'below'
                        if global_id not in event_prev_y:
                            event_prev_y[global_id] = y_center
                            event_first_side[global_id] = current_side
                            event_last_side[global_id] = current_side
                            event_cross_dirs[global_id] = set()
                        else:
                            prev_y = event_prev_y[global_id]
                            if prev_y < LINE_Y <= y_center:
                                event_cross_dirs[global_id].add('down')
                            elif prev_y > LINE_Y >= y_center:
                                event_cross_dirs[global_id].add('up')
                            event_prev_y[global_id] = y_center
                            event_last_side[global_id] = current_side

                    # Person-in-ROI check using center point inside ROI polygon
                    if label_name == 'person':
                        cx = int((box[0] + box[2]) / 2)
                        cy = int((box[1] + box[3]) / 2)
                        if cv2.pointPolygonTest(roi_polygon_cv, (cx, cy), False) >= 0:
                            frame_has_person_in_roi = True

                    draw_tracked_box(frame, box, local_id, label_name, conf, soc)

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
            if processed_offset % output_stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                event_frames_rgb.append(rgb_frame)

        # Reduce per-person crossings to net counts for the event, following rules:
        # - If final side != start side: count 1 in that direction only
        # - If final side == start side and both directions occurred: count 1 up and 1 down
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
            logger.info(f"[{file_basename}] Event kept: person present in ROI for {person_frames_in_roi} frames (>= {PERSON_MIN_FRAMES}).")
            all_clip_frames_rgb.extend(event_frames_rgb)
            # Merge event stats into overall
            unique_objects_detected['person'].update(event_unique_objects['person'])
            unique_objects_detected['car'].update(event_unique_objects['car'])
            persons_up += event_persons_up
            persons_down += event_persons_down
        else:
            logger.info(f"[{file_basename}] Event discarded: person in ROI only {person_frames_in_roi} frames (< {PERSON_MIN_FRAMES}).")
            # Save a representative middle frame for significant-but-no-person events
            if SEND_INSIGNIFICANT_FRAMES:
                try:
                    mid_frame_index = start_frame + (end_frame - start_frame) // 2
                    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        results = object_detection_model.track(frame, imgsz=640, conf=CONF_THRESHOLD, persist=False, verbose=False)
                        if results[0].boxes.id is not None:
                            boxes = results[0].boxes.xyxy.cpu().tolist()
                            clss = results[0].boxes.cls.int().cpu().tolist()
                            confs = results[0].boxes.conf.float().cpu().tolist()
                            class_names = object_detection_model.names

                            local_id_counter = 1
                            for box, cls, conf in zip(boxes, clss, confs):
                                label_name = class_names[cls]
                                draw_tracked_box(frame, box, local_id_counter, label_name, conf, soc)
                                local_id_counter += 1

                        # Save to daily folder inside output dir (YYYYMMDD)
                        date_folder = datetime.now().strftime("%Y%m%d")
                        daily_dir = os.path.join(output_dir, date_folder)
                        os.makedirs(daily_dir, exist_ok=True)
                        frame_filename = f"{input_video_path.split(os.path.sep)[-2][-2:]}H{os.path.splitext(file_basename)[0]}_no_person_{mid_frame_index}.jpg"
                        frame_path = os.path.join(daily_dir, frame_filename)
                        cv2.line(frame, (0, LINE_Y), (orig_w, LINE_Y), COLOR_LINE, 1)
                        cv2.putText(frame, f"Hvirtka Y={LINE_Y}", (10, LINE_Y - 10),
                                    cv2.FONT_HERSHEY_DUPLEX, 1, COLOR_LINE, 1, cv2.LINE_AA)
                        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        #insignificant_motion_frames.append(frame_path) # disabled to keep frames for further analysis only
                        logger.info(f"[{file_basename}] Saved no_person frame to {frame_path}")
                except Exception as e:
                    logger.warning(f"[{file_basename}] Failed to save no_person frame: {e}")

    cap.release()

    if not all_clip_frames_rgb:
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] No significant events with person in ROI found. Full processing took {elapsed_time:.2f} seconds.")
        return {'status': 'no_significant_motion', 'clip_path': None, 'insignificant_frames': insignificant_motion_frames}

    final_clip = ImageSequenceClip(all_clip_frames_rgb, fps=fps)
    output_filename = os.path.join(output_dir, file_basename)
    logger.info(f"[{file_basename}] Writing final highlight clip to {output_filename}...")
    final_clip.write_videofile(
        output_filename, codec='libx264', audio=False, bitrate='2000k',
        preset='medium', threads=4, logger=None
    )

    file_duration = final_clip.duration
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
