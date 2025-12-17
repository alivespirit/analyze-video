import os
import datetime
import asyncio
import concurrent.futures
import logging
import random
import sys
import time
import json
import cv2
import numpy as np
import psutil

from dotenv import load_dotenv
from telegram import InlineKeyboardMarkup, InlineKeyboardButton, InputMediaPhoto
import telegram.error
from telegram.ext import Application, CallbackQueryHandler
from telegram.helpers import escape_markdown
from google import genai
from google.genai import types

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

from logging.handlers import TimedRotatingFileHandler

# Importing ImageSequenceClip from moviepy for MP4 extraction
from moviepy import ImageSequenceClip

# --- NEW: Import YOLO for object detection ---
from ultralytics import YOLO

# --- NEW: Import teslapy for Tesla integration ---
try:
    import teslapy
except ImportError:
    print("teslapy is not installed. Tesla integration will be disabled.")
    teslapy = None
# -------------------------------------------

# --- State for Grouping "No Motion" Messages ---
# These are safe to use as globals because the executor has max_workers=1,
# ensuring sequential processing and preventing race conditions.
no_motion_group_message_id = None
no_motion_grouped_videos = []

# --- NEW: Add a lock for Telegram message grouping ---
telegram_lock = asyncio.Lock()

RESTART_REQUESTED = False
MAIN_SCRIPT_PATH = os.path.abspath(__file__)

class MainScriptChangeHandler(FileSystemEventHandler):
    def __init__(self, stop_event_ref, script_path_ref):
        self.stop_event = stop_event_ref
        self.script_path = script_path_ref
        self.triggered_restart = False # Ensure we only trigger once

    def on_modified(self, event):
        """
        Handles the file modification event from watchdog.
        If the modified file is the main script, it triggers a graceful restart.
        """
        if self.triggered_restart:
            return

        # Watchdog might trigger for .pyc or other related files if watching a directory
        # So explicitly check if the main script itself was modified
        if os.path.abspath(event.src_path) == self.script_path:
            logger.info(f"Detected change in {self.script_path}. Initiating graceful restart...")
            global RESTART_REQUESTED
            RESTART_REQUESTED = True
            self.stop_event.set() # Signal all other components to stop
            self.triggered_restart = True # Prevent further triggers from this handler instance

load_dotenv()  ## load all the environment variables

LOG_PATH = os.getenv("LOG_PATH", default="")

class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def doRollover(self):
        """
        Overrides the default rollover to rename log files with a custom format.
        e.g., from `video_processor.log.2025-05-25` to `video_processor_2025-05-25.log`.
        """
        super().doRollover()
        # Find the most recent rotated file
        dirname, basename = os.path.split(self.baseFilename)
        for filename in os.listdir(dirname or "."):
            if filename.startswith(basename + "."):
                # e.g., video_processor.log.2025-05-25
                old_path = os.path.join(dirname, filename)
                # e.g., video_processor_2025-05-25.log
                new_name = filename.replace(".log.", "_") + ".log"
                new_path = os.path.join(dirname, new_name)
                if not os.path.exists(new_path):
                    os.rename(old_path, new_path)

# Custom filter to suppress stacktraces for network errors while keeping the error messages
class NetworkErrorFilter(logging.Filter):
    def filter(self, record):
        """
        Suppresses stack traces for common, non-critical network errors.
        This keeps the logs cleaner by showing the error message without the full traceback.
        """
        # Only suppress stacktraces (exc_info) for network-related errors, keep the message
        if record.levelname == 'ERROR' and record.exc_info:
            message = record.getMessage()
            if any(phrase in message for phrase in [
                'Exception happened while polling for updates',
                'getaddrinfo failed',
                'ConnectError',
                'NetworkError'
            ]):
                # Keep the error message but remove the stacktrace
                record.exc_info = None
                record.exc_text = None
                record.stack_info = None
        return True

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_file = os.path.join(LOG_PATH, "video_processor.log")  # No date in filename; handler adds it
file_handler = CustomTimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=30, encoding='utf8', utc=False
)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO)

# --- FIX: Apply the filter directly to the handlers ---
network_filter = NetworkErrorFilter()
file_handler.addFilter(network_filter)
console_handler.addFilter(network_filter)

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set root logger level
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# ---------------------

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("moviepy").setLevel(logging.WARNING) # Keep moviepy logs quiet
logging.getLogger("ultralytics").setLevel(logging.WARNING) # Suppress YOLO logs

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
USERNAME = os.getenv("TELEGRAM_NOTIFY_USERNAME")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER")
OBJECT_DETECTION_MODEL_PATH = os.getenv("OBJECT_DETECTION_MODEL_PATH", default="best_openvino_model")
TESLA_EMAIL = os.getenv("TESLA_EMAIL")
TESLA_REFRESH_TOKEN = os.getenv("TESLA_REFRESH_TOKEN")

# --- Define and create a local temporary directory ---
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(MAIN_SCRIPT_PATH) # MAIN_SCRIPT_PATH is already defined
# Define the path for the temp subfolder
TEMP_DIR = os.path.join(SCRIPT_DIR, "temp")
# Create the temp directory if it doesn't exist; this is safe to run every time.
try:
    os.makedirs(TEMP_DIR, exist_ok=True)
    logger.info(f"Temporary file directory ensured at: {TEMP_DIR}")
except OSError as e:
    logger.critical(f"Could not create temporary directory at {TEMP_DIR}: {e}", exc_info=True)
    exit(1) # Can't proceed without a temp dir, so exit.
# ---------------------------------------------

# --- Motion detection configuration ---
MIN_CONTOUR_AREA = 1800         # Minimum area of a contour to be considered significant
ROI_CONFIG_FILE = os.path.join(SCRIPT_DIR, "roi.json")
PADDING_SECONDS = 1.5
# ### WARM-UP CONFIGURATION ###
# Number of initial frames to ignore while the background model stabilizes.
# A value of 25-50 is usually good for a 25fps video (1-2 seconds).
WARMUP_FRAMES = 15
# ### CONFIGURATION FOR EVENT GROUPING ###
# If motion stops for more than this many seconds, start a new clip segment.
MAX_EVENT_GAP_SECONDS = 3.0
# ### DURATION FILTER CONFIGURATION ###
# Discard any motion event that lasts for less than this many seconds.
MIN_EVENT_DURATION_SECONDS = 1.0
# From insignificant motions that last longer than this, a frame will be extracted
MIN_INSIGNIFICANT_EVENT_DURATION_SECONDS = 0.2
# ### PERFORMANCE CONFIGURATION ###
CROP_PADDING = 30  # Pixels to add around the ROI bounding box for safety
# ### SANITY CHECK CONFIGURATION ###
# If a bounding box is larger than this percentage of the total analysis area,
# discard it as it's likely a noise-polluted first frame.
MAX_BOX_AREA_PERCENT = 0.80

# --- NEW: Tesla SoC file ---
TESLA_SOC_FILE = os.path.join(SCRIPT_DIR, "tesla_soc.txt")
TESLA_LAST_CHECK = 0  # Timestamp of the last SoC check
if not teslapy or not TESLA_REFRESH_TOKEN or not TESLA_EMAIL:
    logger.info(f"Tesla integration not configured (teslapy or TESLA_REFRESH_TOKEN or TESLA_EMAIL missing). Tesla SoC checks disabled.")
    TESLA_SOC_CHECK_ENABLED = False
else:
    TESLA_SOC_CHECK_ENABLED = True

# --- NEW: Object Detection Configuration ---
CONF_THRESHOLD = 0.45
LINE_Y = 860  # Counting Line Y-Coordinate
COLOR_PERSON = (100, 200, 0)
COLOR_CAR = (200, 120, 0)
COLOR_DEFAULT = (255, 255, 255)
COLOR_LINE = (0, 255, 255)
# -----------------------------------------

NO_ACTION_RESPONSES = [
    "Нема шо дивитись",
    "Ніц цікавого",
    "Йойки, геть ніц",
    "Все спокійно",
    "Німа нічо",
    "Журбинка якась",
    "Сумулька лиш",
    "Чортівні нема",
    "Всьо чотко",
    "Геть нема екшину"
]

# --- Check Environment Variables ---
if not all([GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID, USERNAME, VIDEO_FOLDER]):
    logger.critical("ERROR: One or more essential environment variables are missing. Exiting.")
    exit(1) # Exit if critical env vars are missing
if not os.path.isdir(VIDEO_FOLDER):
    logger.critical(f"ERROR: VIDEO_FOLDER '{VIDEO_FOLDER}' does not exist or is not a directory. Exiting.")
    exit(1)
if not os.path.exists(OBJECT_DETECTION_MODEL_PATH):
    logger.critical(f"ERROR: OBJECT_DETECTION_MODEL_PATH '{OBJECT_DETECTION_MODEL_PATH}' does not exist. Exiting.")
    exit(1)
# -----------------------------------

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini configured successfully.")
except Exception as e:
    logger.critical(f"Failed to configure Gemini: {e}", exc_info=True) # Use exc_info for traceback
    exit(1)

# --- NEW: Load Object Detection Model ---
try:
    object_detection_model = YOLO(OBJECT_DETECTION_MODEL_PATH, task='detect')
    logger.info(f"Object detection model loaded successfully from {OBJECT_DETECTION_MODEL_PATH}.")
except Exception as e:
    logger.critical(f"Failed to load object detection model: {e}", exc_info=True)
    exit(1)
# ------------------------------------


# Initialize the Application
try:
    application = Application.builder() \
        .token(TELEGRAM_TOKEN) \
        .http_version("1.1") \
        .get_updates_http_version("1.1") \
        .connection_pool_size(32) \
        .pool_timeout(60) \
        .read_timeout(60) \
        .write_timeout(60) \
        .build()
    logger.info("Telegram Application built successfully.")
except Exception as e:
     logger.critical(f"Failed to build Telegram Application: {e}", exc_info=True)
     exit(1)

# Thread Pool Executor for Blocking Tasks
# We limit the pool to 1 worker. This effectively creates a processing queue,
# preventing multiple CPU-heavy 'detect_motion' tasks from running simultaneously
# and overwhelming the system.
max_workers = 1 
try:
    # A dedicated executor for CPU-bound motion detection to ensure it runs one at a time.
    motion_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"ThreadPoolExecutor for motion detection initialized with a single worker.")
    
    # A general-purpose executor for I/O-bound tasks like Gemini calls.
    io_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"ThreadPoolExecutor for I/O tasks initialized with a single worker.")
except Exception as e:
     logger.critical(f"Failed to initialize ThreadPoolExecutors: {e}", exc_info=True)
     exit(1)


def load_tesla_soc(filepath):
    """Loads the Tesla SoC from the cache file if it exists."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            soc = int(f.read().strip())
            return soc
    return None

# --- NEW: Tesla SoC Check Function ---
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
        if (current_time - last_modified_time < 7200) or (current_time - TESLA_LAST_CHECK < 600):  # 2 hours or cooldown 10 minutes
            if soc is not None:
                logger.info(f"[{file_basename}] Using cached Tesla SoC: {soc}%")
                return soc
            else:
                logger.warning(f"[{file_basename}] Could not read cached SoC file. Will fetch fresh data.")

    logger.info(f"[{file_basename}] Fetching fresh Tesla SoC from API...")
    try:
        with teslapy.Tesla(email=TESLA_EMAIL) as tesla:
            if not tesla.authorized:
                tesla.refresh_token(refresh_token=TESLA_REFRESH_TOKEN)
            vehicles = tesla.vehicle_list()
            if not vehicles:
                logger.error(f"[{file_basename}] No vehicles found in Tesla account.")
                return None
            
            # Use the first vehicle in the list
            vehicle = vehicles[0]
            
            # Get charge state data
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

# --- ROI Loading Function ---
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
    
    if label_name == 'person': color = COLOR_PERSON
    elif label_name == 'car': color = COLOR_CAR
    else: color = COLOR_DEFAULT

    label_text = f"{label_name} {local_id} {conf:.0%}"

    if TESLA_SOC_CHECK_ENABLED:
        # Tesla location check
        tx, ty = 1150, 450
        
        # Check if this object is a CAR and if the point is INSIDE the box
        if label_name == 'car':
            if x1 <= tx <= x2 and y1 <= ty <= y2:
                # Overwrite the label text with Tesla SoC info
                if soc is not None:
                    label_text = f"Tesla {conf:.0%} / SoC {soc}%"
                else:
                    label_text = f"Tesla {conf:.0%}"
    # ------------------------------------------------

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 1, 1)
    
    cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)
    cv2.putText(frame, label_text, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

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
    
    # ### DEFINE THE CROP BOX BASED ON THE ROI ###
    x_coords = roi_poly_points[:, 0]
    y_coords = roi_poly_points[:, 1]
    crop_x1 = max(0, np.min(x_coords) - CROP_PADDING)
    crop_y1 = max(0, np.min(y_coords) - CROP_PADDING)
    crop_x2 = min(orig_w, np.max(x_coords) + CROP_PADDING)
    crop_y2 = min(orig_h, np.max(y_coords) + CROP_PADDING)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    logger.info(f"[{file_basename}] Original frame: {orig_w}x{orig_h}. Analyzing cropped region: {crop_w}x{crop_h} at ({crop_x1},{crop_y1}).")

    # ### MODIFIED: ALL SUBSEQUENT SETUP IS RELATIVE TO THE CROP ###
    local_roi_points = roi_poly_points.copy()
    local_roi_points[:, 0] -= crop_x1
    local_roi_points[:, 1] -= crop_y1
    
    analysis_roi_points = local_roi_points.astype(np.int32)
    
    backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=800, detectShadows=True)
    roi_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [analysis_roi_points], 255)

    # --- MODIFIED: Smarter background model pre-training ---
    logger.info(f"[{file_basename}] Starting smart background model pre-training...")
    pre_trained = False
    # Define candidate start times in seconds for pre-training (e.g., 30s, 40s, 50s, 20s)
    training_candidate_times = [30, 40, 50, 20]
    frames_to_sample = 25  # ~1 second to check for motion
    frames_to_train = 150  # ~6 seconds to train the model

    for start_sec in training_candidate_times:
        start_frame = int(start_sec * fps)
        if total_frames < start_frame + frames_to_sample:
            continue # Not enough frames for this candidate

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Pre-check for motion in this segment
        motion_detected_in_segment = False
        temp_backSub = cv2.createBackgroundSubtractorKNN(history=50, dist2Threshold=800, detectShadows=True)
        
        for i in range(frames_to_sample):
            ret, frame = cap.read()
            if not ret: break
            
            cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
            fg_mask = temp_backSub.apply(roi_frame)
            
            # A simple check for significant contours
            if i > 5 and cv2.countNonZero(fg_mask) > (roi_mask.size * 0.1): # if >10% of ROI has motion
                motion_detected_in_segment = True
                logger.info(f"[{file_basename}] Motion detected in pre-training candidate segment at {start_sec}s. Trying next segment.")
                break
        
        if not motion_detected_in_segment:
            logger.info(f"[{file_basename}] Found a static segment at {start_sec}s. Pre-training background model...")
            # The first 25 frames are already read, continue reading for the full training duration
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            for _ in range(frames_to_train):
                ret, frame = cap.read()
                if not ret: break
                
                cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
                backSub.apply(roi_frame)
            
            pre_trained = True
            break # Exit after successful training

    if pre_trained:
        logger.info(f"[{file_basename}] Background model pre-trained. Resetting to start for analysis.")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        logger.warning(f"[{file_basename}] Could not find a static segment to pre-train model. Using standard warm-up.")
    
    # If we pre-trained, we only need a few frames for camera stabilization. Otherwise, use original WARMUP.
    EFFECTIVE_WARMUP = 5 if pre_trained else WARMUP_FRAMES
    
    motion_events = []
    for frame_index in range(EFFECTIVE_WARMUP, total_frames):
        ret, frame = cap.read()
        if not ret: break

        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
        fg_mask = backSub.apply(roi_frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # --- MODIFIED: Check for large contours but don't filter by solidity here ---
        # The object detector is more reliable than shape analysis.
        # We still check for a minimum area to avoid noise triggering the expensive detector.
        large_enough_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_CONTOUR_AREA]
        if large_enough_contours:
            motion_events.append((frame_index, large_enough_contours))

    if not motion_events:
        logger.info(f"[{file_basename}] No significant motion found.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        return {'status': 'no_motion', 'clip_path': None, 'insignificant_frames': []}

    # --- MODIFIED: Simplified motion event processing ---
    # We no longer create union_rects or interpolate. We just need the frame ranges.
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

    # --- NEW: Cooldown configuration for line crossing ---
    LINE_CROSSING_COOLDOWN_SECONDS = 2.0
    # ----------------------------------------------------

    # --- Check Tesla SoC once per video processing ---
    soc = check_tesla_soc(file_basename) if TESLA_SOC_CHECK_ENABLED else None

    logger.info(f"[{file_basename}] Found {len(sub_clips)} raw motion event(s). Filtering by duration...")
    significant_sub_clips = []
    insignificant_motion_frames = []
    for start_frame, end_frame in sub_clips:
        duration_frames = end_frame - start_frame
        duration_seconds = duration_frames / fps
        if duration_seconds >= MIN_EVENT_DURATION_SECONDS:
            logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is SIGNIFICANT. Will process with tracker.")
            significant_sub_clips.append((start_frame, end_frame))
        elif duration_seconds >= MIN_INSIGNIFICANT_EVENT_DURATION_SECONDS:
            logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is insignificant. Extracting frame.")
            mid_frame_index = start_frame + (end_frame - start_frame) // 2
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_index)
            ret, frame = cap.read()
            if ret:
                # Run detection on the single frame
                results = object_detection_model.track(frame, imgsz=640, conf=CONF_THRESHOLD, persist=False, verbose=False)
                if results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().tolist()
                    clss = results[0].boxes.cls.int().cpu().tolist()
                    confs = results[0].boxes.conf.float().cpu().tolist()
                    class_names = object_detection_model.names
                    
                    # Use a simple counter for local ID on single frames
                    local_id_counter = 1
                    for box, cls, conf in zip(boxes, clss, confs):
                        label_name = class_names[cls]
                        draw_tracked_box(frame, box, local_id_counter, label_name, conf, soc)
                        local_id_counter += 1

                frame_filename = f"{os.path.splitext(file_basename)[0]}_insignificant_{mid_frame_index}.jpg"
                frame_path = os.path.join(output_dir, frame_filename)
                # Save with compression
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                insignificant_motion_frames.append(frame_path)
                logger.info(f"[{file_basename}] Saved insignificant motion frame to {frame_path}")
        else:
            logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is too short. Discarding as noise/shadow.")
    
    if not significant_sub_clips:
        logger.info(f"[{file_basename}] No significant long-duration motion found.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        return {'status': 'no_significant_motion', 'clip_path': None, 'insignificant_frames': insignificant_motion_frames}

    # --- NEW: Object Tracking on Significant Clips ---
    logger.info(f"[{file_basename}] Starting object tracking for {len(significant_sub_clips)} significant event(s).")
    
    # Reset tracker state for the new video
    if hasattr(object_detection_model, 'predictor') and object_detection_model.predictor is not None:
        if hasattr(object_detection_model.predictor, 'trackers') and object_detection_model.predictor.trackers:
            object_detection_model.predictor.trackers[0].reset()
            logger.info(f"[{file_basename}] Object tracker state reset.")

    all_clip_frames_rgb = []
    class_names = object_detection_model.names
    id_mapping = {}
    class_counters = {}
    unique_objects_detected = {'person': set(), 'car': set()}
    previous_positions = {}
    persons_up = 0
    persons_down = 0
    line_crossing_cooldown = {} # {global_id: cooldown_end_frame}

    # Get all frame indices that need processing by the tracker
    frames_to_process_indices = set()
    for clip_index, (start_frame, end_frame) in enumerate(significant_sub_clips):
        duration_seconds = (end_frame - start_frame) / fps
        is_long_motion = duration_seconds > 4.0
        padding_seconds_adjusted = 0.5 if is_long_motion else PADDING_SECONDS
        padded_start = max(0, start_frame - int(padding_seconds_adjusted * fps))
        padded_end = min(total_frames, end_frame + int(padding_seconds_adjusted * fps))

        # NEW: If first motion is at the very start, include from frame 0
        if clip_index == 0 and start_frame <= (EFFECTIVE_WARMUP + fps * 1.0):
            logger.info(f"[{file_basename}] Motion starts at the beginning. Including video from frame 0.")
            padded_start = 0

        # NEW: Check if this clip should be sped up
        if is_long_motion:
            logger.info(f"[{file_basename}] Long motion event ({duration_seconds:.2f}s) detected. Speeding up clip segment.")

        for i in range(padded_start, padded_end):
            if is_long_motion and (i - padded_start) % 2 != 0:
                continue  # Skip every other frame for long motions
            frames_to_process_indices.add(i)
    
    sorted_frame_indices = sorted(list(frames_to_process_indices))

    for frame_idx in sorted_frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: continue

        # Run Tracking
        results = object_detection_model.track(frame, imgsz=640, conf=CONF_THRESHOLD, persist=True, verbose=False, tracker="bytetrack.yaml")

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().tolist()
            global_ids = results[0].boxes.id.int().cpu().tolist()
            clss = results[0].boxes.cls.int().cpu().tolist()
            confs = results[0].boxes.conf.float().cpu().tolist()

            for box, global_id, cls, conf in zip(boxes, global_ids, clss, confs):
                label_name = class_names[cls]

                if label_name not in class_counters: class_counters[label_name] = 1
                if global_id not in id_mapping:
                    id_mapping[global_id] = class_counters[label_name]
                    class_counters[label_name] += 1
                local_id = id_mapping[global_id]

                if label_name in unique_objects_detected:
                    unique_objects_detected[label_name].add(global_id)

                y_center = int((box[1] + box[3]) / 2)
                if global_id in previous_positions:
                    prev_y = previous_positions[global_id]
                    if label_name == 'person':
                        # Check if cooldown has passed for this person
                        if frame_idx >= line_crossing_cooldown.get(global_id, 0):
                            crossed = False
                            if prev_y < LINE_Y <= y_center:
                                persons_down += 1
                                crossed = True
                            elif prev_y > LINE_Y >= y_center:
                                persons_up += 1
                                crossed = True
                            
                            if crossed:
                                # Set cooldown end frame
                                cooldown_frames = int(LINE_CROSSING_COOLDOWN_SECONDS * fps)
                                line_crossing_cooldown[global_id] = frame_idx + cooldown_frames
                previous_positions[global_id] = y_center

                draw_tracked_box(frame, box, local_id, label_name, conf, soc)
        
        # Draw crossing line on frame
        cv2.line(frame, (0, LINE_Y), (orig_w, LINE_Y), COLOR_LINE, 1)
        cv2.putText(frame, f"Hvirtka Y={LINE_Y}", (10, LINE_Y - 10), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, COLOR_LINE, 1, cv2.LINE_AA)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        all_clip_frames_rgb.append(rgb_frame)

    cap.release()

    if not all_clip_frames_rgb:
        logger.error(f"[{file_basename}] No frames collected for the clip after tracking.")
        return {'status': 'error', 'clip_path': None, 'insignificant_frames': insignificant_motion_frames}
        
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
    
    # --- Final result preparation ---
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


def analyze_video(motion_result, video_path):
    """
    Generates a descriptive text for a video using Google's Gemini AI and formats the final response.

    This function runs in the I/O executor. Its behavior depends on the motion detection result:
    -   **No Motion**: Returns a random "nothing to see" message.
    -   **Gate Crossing**: Returns a formatted message indicating the crossing event, skipping Gemini.
    -   **Off-Peak Hours**: Skips Gemini and returns a summary based on object counts to save API calls.
    -   **Significant Motion (Peak Hours)**: Sends the highlight clip to Gemini for analysis.
    -   **Error/Insignificant Motion (Peak Hours)**: Sends the full video to Gemini for analysis.

    It handles model selection (Pro vs. Flash), API retries, and error handling.

    Args:
        motion_result (dict): The result dictionary from the `detect_motion` function.
        video_path (str): The path to the original video file.

    Returns:
        dict: A dictionary containing the formatted text response, paths to any insignificant frames,
              and the path to the highlight clip if one was generated.
    """
    file_basename = os.path.basename(video_path)
    timestamp = f"_{video_path.split(os.path.sep)[-2][-2:]}H{file_basename[:3]}:_ "
    use_files_api = False
    now = datetime.datetime.now()

    # Handle case where motion detection fails
    if motion_result is None or not isinstance(motion_result, dict):
        logger.warning(f"[{file_basename}] Motion detection returned an unexpected value: {motion_result}. Analyzing full video.")
        motion_result = {'status': 'error', 'clip_path': None, 'insignificant_frames': []}

    detected_motion_status = motion_result['status']
    
    # --- Skip Gemini analysis for no motion events ---
    if detected_motion_status == "no_motion":
        logger.info(f"[{file_basename}] Skipping Gemini analysis (no motion).")
        return {'response': timestamp + "\u2714\uFE0F " + random.choice(NO_ACTION_RESPONSES), 'insignificant_frames': [], 'clip_path': None}

    # --- Skip Gemini analysis for gate crossing events ---
    if detected_motion_status == "gate_crossing":
        logger.info(f"[{file_basename}] Skipping Gemini analysis (gate crossing).")
        direction = motion_result.get('crossing_direction')
        persons_up = motion_result.get('persons_up', 0)
        persons_down = motion_result.get('persons_down', 0)
        
        direction_text = ""
        if direction == 'up':
            direction_text = "\U0001F6A7" +"\U0001F6B6\u200D\u27A1\uFE0F" * persons_up 
        elif direction == 'down':
            direction_text = "\U0001F6B6\u200D\u27A1\uFE0F" * persons_down + " \U0001F6A7"
        elif direction == 'both':
            direction_text = "\U0001F6B6\u200D\u27A1\uFE0F" * persons_down + " \U0001F6A7" + "\U0001F6B6\u200D\u27A1\uFE0F" * persons_up

        analysis_result = direction_text
        if 9 <= now.hour <= 13:
            analysis_result += f"\n{USERNAME}"
        return {
            'response': timestamp + analysis_result,
            'insignificant_frames': motion_result.get('insignificant_frames', []),
            'clip_path': motion_result.get('clip_path')
        }
    # -----------------------------------------

    # --- Skip Gemini analysis during off-peak hours to keep under rate limits ---
    if now.hour < 9 or now.hour > 18:
        logger.info(f"[{file_basename}] Skipping Gemini analysis (off-peak hours).")
        if detected_motion_status == "error":
            return {'response': timestamp + "\U0001F4A2 Шось неясно", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': None}
        elif detected_motion_status == "no_significant_motion":
            return {'response': timestamp + "\U0001F518 Шось там тойво...", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': None}
        elif detected_motion_status == "significant_motion":
            # --- MODIFIED: Append detected object counts to the off-peak message ---
            persons = motion_result.get('persons_detected', 0)
            cars = motion_result.get('cars_detected', 0)
            details = []
            if persons > 0:
                details.append(f"{persons} \U0001F9CD")
            if cars > 0:
                details.append(f"{cars} \U0001F699")
            
            if details:
                return {'response': timestamp + f"\u2611\uFE0F Шось там {', '.join(details)}", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
            else:
                return {'response': timestamp + "\u2611\uFE0F Виявлено капець рух.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}

    # Analyze with Gemini
    video_to_process = None
    video_bytes_obj = None
    if detected_motion_status == "error":
        logger.warning(f"[{file_basename}] Error during motion detection. Analyzing full video.")
        video_to_process = video_path
    elif detected_motion_status == "no_significant_motion":
        logger.info(f"[{file_basename}] Analyzing full video as there was no significant motion.")
        video_to_process = video_path
    elif detected_motion_status == "significant_motion":
        logger.info(f"[{file_basename}] Running Gemini analysis for detected motion at {motion_result['clip_path']}")
        video_to_process = motion_result['clip_path']
    
    # If there's nothing to process (e.g., no_significant_motion but we decide not to analyze full video)
    if not video_to_process:
         logger.info(f"[{file_basename}] No video to analyze, but insignificant frames may exist.")
         return {'response': timestamp + "\U0001F4A2 Нема значного руху.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': None}

    try:
        if use_files_api:
            video_bytes_obj = client.files.upload(file=video_to_process)
            # Wait up to 2 minutes (120 seconds) for video processing
            max_wait_seconds = 120
            wait_interval = 10
            waited = 0
            while video_bytes_obj.state == "PROCESSING":
                if waited >= max_wait_seconds:
                    logger.error(f"[{file_basename}] Video processing timed out after {max_wait_seconds} seconds.")
                    return {'response': timestamp + "\u274C Відео не вдалося обробити (timeout).", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
                logger.info(f"[{file_basename}] Waiting for video to be processed ({waited}/{max_wait_seconds}s).")
                time.sleep(wait_interval)
                waited += wait_interval
                video_bytes_obj = client.files.get(name=video_bytes_obj.name)

            if video_bytes_obj.state == "FAILED":
                logger.error(f"[{file_basename}] Video processing failed: {video_bytes_obj.error_message}")
                return {'response': timestamp + "\u274C Відео не вдалося обробити.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
        else:
            try:
                with open(video_to_process, 'rb') as f:
                    video_data = f.read()
            except Exception as e:
                logger.error(f"[{file_basename}] Error reading video file: {e}")
                return {'response': timestamp + "\u274C Відео не вдалося прочитати.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}

        pro_model_file = os.path.join(SCRIPT_DIR, "model_pro")
        if os.path.exists(pro_model_file) and (9 <= now.hour <= 13):
            # If it's between 9:00 and 13:59, use the Pro model
            with open(pro_model_file, "r", encoding="utf-8") as pro_model:
                model_main = pro_model.read().strip()
            model_fallback = 'gemini-2.5-flash'
            model_fallback_text = '_[2.5F]_ '
        else:
            # Outside of that time, use the Flash models
            model_main = 'gemini-2.5-flash'
            model_fallback = 'gemini-2.5-flash-lite'
            model_fallback_text = '_[2.5FL]_ '

        # Make final fallback optional
        final_fallback_model_file = os.path.join(SCRIPT_DIR, "model_final_fallback")
        final_fallback_enabled = os.path.exists(final_fallback_model_file)
        if final_fallback_enabled:
            with open(final_fallback_model_file, "r", encoding="utf-8") as final_fallback_model:
                model_final_fallback = final_fallback_model.read().strip()
        model_final_fallback_text = '_[FF]_ '

        sampling_rate = 5  # sampling rate in FPS, valid only for inline_data
        max_retries = 3

        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
                prompt = prompt_file.read().strip()
            logger.debug(f"[{file_basename}] Prompt loaded successfully from {prompt_file_path}.")
        except FileNotFoundError:
            logger.error(f"[{file_basename}] Prompt file not found: {prompt_file_path}")
            return {'response': timestamp + "Prompt file not found.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
        except Exception as e:
            logger.error(f"[{file_basename}] Error reading prompt file: {e}", exc_info=True)
            return {'response': timestamp + "Error reading prompt file.", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}

        if use_files_api:
            contents = [video_bytes_obj, prompt]
        else:
            contents = types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=video_data,
                            mime_type='video/mp4'
                        ),
                        video_metadata=types.VideoMetadata(fps=sampling_rate)
                    ),
                    types.Part(text=prompt)
                ]
            )

        analysis_result = ""
        additional_text = ""

        for attempt in range(max_retries):
            try:
                logger.info(f"[{file_basename}] Generating content ({model_main}), attempt {attempt+1}...")
                response = client.models.generate_content(
                              model=model_main,
                              contents=contents,
                              config=types.GenerateContentConfig(
                                  automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                      disable=True
                                  )
                              )
                          )
                logger.info(f"[{file_basename}] {model_main} response received.")
                if response.text is None or response.text.strip() == "":
                    raise ValueError(f"{model_main} returned an empty response with reason {response.candidates[0].finish_reason.name}.")
                analysis_result = response.text
                break
            except Exception as e_main:
                try:
                    logger.warning(f"[{file_basename}] {model_main} failed. Falling back to {model_fallback}. Message: {e_main}")
                    response = client.models.generate_content(
                                  model=model_fallback,
                                  contents=contents,
                                  config=types.GenerateContentConfig(
                                      automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                          disable=True
                                      )
                                  )
                              )
                    logger.info(f"[{file_basename}] {model_fallback} response received.")
                    if response.text is None or response.text.strip() == "":
                        raise ValueError(f"{model_fallback} returned an empty response with reason {response.candidates[0].finish_reason.name}.")
                    analysis_result = model_fallback_text + response.text
                    break
                except Exception as e_fallback:
                    logger.warning(f"[{file_basename}] {model_fallback} also failed: {e_fallback}")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s
                        logger.warning(f"[{file_basename}] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        if final_fallback_enabled:
                            try:
                                logger.info(f"[{file_basename}] Attempting {model_final_fallback} as final fallback...")
                                response = client.models.generate_content(
                                              model=model_final_fallback,
                                              contents=contents,
                                              config=types.GenerateContentConfig(
                                                  automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                                      disable=True
                                                  )
                                              )
                                          )
                                logger.info(f"[{file_basename}] {model_final_fallback} response received.")
                                analysis_result = model_final_fallback_text + response.text
                                break
                            except Exception as e_fallback_final:
                                logger.error(f"[{file_basename}] {model_final_fallback} failed as well: {e_fallback_final}")
                                logger.error(f"[{file_basename}] Giving up after retries.")
                                raise # Re-raise the exception to handle it in the outer scope
                        else:
                            logger.error(f"[{file_basename}] Giving up after retries.")
                            raise # Re-raise the exception to handle it in the outer scope

        analysis_result = (analysis_result[:512] + '...') if len(analysis_result) > 1023 else analysis_result

        logger.info(f"[{file_basename}] Response: {analysis_result}")

        # --- MODIFIED: Append detected object counts to the Gemini result ---
        persons = motion_result.get('persons_detected', 0)
        cars = motion_result.get('cars_detected', 0)
        details = []
        if persons > 0:
            details.append(f"{persons} \U0001F9CD")
        if cars > 0:
            details.append(f"{cars} \U0001F699")
        
        if details:
            analysis_result += f" ({', '.join(details)})"
        # ----------------------------------------------------------------

        # Notify username if needed (disabled to have notifications only on gate crossings)
        #if detected_motion_status == "significant_motion" and (9 <= now.hour <= 13):
        #    additional_text += f"\n{USERNAME}"

        if detected_motion_status == "significant_motion":
            timestamp += "\u2705 *Отакої!* "
        else:
            timestamp += "\u2747\uFE0F "

        return {'response': timestamp + analysis_result + additional_text, 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}

    except Exception as e_analysis:
        logger.error(f"[{file_basename}] Video analysis failed: {e_analysis}", exc_info=False)
        if '429' in str(e_analysis):
            return {'response': timestamp + "\u26A0\uFE0F Ти забагато питав...", 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
        else:
            return {'response': timestamp + "\u274C Відео не вдалося проаналізувати: " + str(e_analysis)[:512] + '...', 'insignificant_frames': motion_result['insignificant_frames'], 'clip_path': motion_result.get('clip_path')}
    finally:
        # This block executes after the try/except block, ensuring cleanup happens.
        if use_files_api and 'video_bytes_obj' in locals() and hasattr(video_bytes_obj, 'name'):
            try:
                client.files.delete(name=video_bytes_obj.name)
                logger.info(f"[{file_basename}] Successfully deleted uploaded file from Gemini API.")
            except Exception as e_delete:
                logger.warning(f"[{file_basename}] Failed to delete uploaded file from Gemini API: {e_delete}", exc_info=False)


# --- FileHandler (uses executor) ---
class FileHandler(FileSystemEventHandler):
    def __init__(self, loop, app):
        self.loop = loop
        self.app = app
        self.logger = logging.getLogger(__name__)
        self.telegram_lock = telegram_lock

    def on_created(self, event):
        """
        Handles the 'file created' event from watchdog for .mp4 files.
        Schedules the `handle_event` coroutine to run on the event loop.

        Args:
            event (watchdog.events.FileSystemEvent): The event object from watchdog.
        """
        if event.is_directory: return
        if not event.src_path.endswith('.mp4'): return

        coro = self.handle_event(event)
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            self.logger.warning(f"Event loop not running when trying to schedule handler for {event.src_path}")

    async def handle_event(self, event):
        """
        The core asynchronous handler for processing a new video file.

        This function orchestrates the entire pipeline for a single video:
        1.  Waits for the file to be fully written to disk.
        2.  Schedules `detect_motion` (CPU-bound) in its dedicated executor.
        3.  Schedules `analyze_video` (I/O-bound) in its dedicated executor.
        4.  Acquires a lock to safely interact with Telegram.
        5.  Sends the results (animation, text, photos) to the Telegram chat.
        6.  Groups messages for insignificant motion to avoid spam.
        7.  Cleans up temporary files (highlight clips, frames).

        Args:
            event (watchdog.events.FileSystemEvent): The event object from watchdog.
        """
        # This function is replaced with the new logic
        file_path = event.src_path
        file_basename = os.path.basename(file_path)
        timestamp_text = f"{file_path.split(os.path.sep)[-2][-2:]}H{file_basename[:3]}"
        self.logger.info(f"[{file_basename}] New file detected: {file_path}")

        try:
            await self.wait_for_file_stable(file_path, file_basename)
        except FileNotFoundError:
            self.logger.warning(f"[{file_basename}] File disappeared before analysis could start.")
            return
        except Exception as e_wait:
            self.logger.error(f"[{file_basename}] Error waiting for file stability: {e_wait}", exc_info=True)
            return

        sent_message = None # Will store the message object to reply to

        try:
            current_loop = asyncio.get_running_loop()
            
            # --- REFACTORED: Run motion detection and Gemini analysis in separate executors ---
            # 1. Run CPU-bound motion detection in the single-worker executor.
            self.logger.info(f"[{file_basename}] Queuing motion detection...")
            motion_result = await current_loop.run_in_executor(
                motion_executor, detect_motion, file_path, TEMP_DIR
            )
            self.logger.info(f"[{file_basename}] Motion detection complete. Status: {motion_result.get('status')}")

            # 2. Run I/O-bound Gemini analysis in the multi-worker executor.
            # This can run in parallel with the next video's motion detection.
            self.logger.info(f"[{file_basename}] Queuing motion analysis...")
            analysis_result = await current_loop.run_in_executor(
                io_executor, analyze_video, motion_result, file_path
            )
            
            video_response = analysis_result['response']
            insignificant_frames = analysis_result['insignificant_frames']
            clip_path = analysis_result.get('clip_path')
            self.logger.info(f"[{file_basename}] Analysis complete.")
        except Exception as e:
            self.logger.error(f"[{file_basename}] Error during video processing pipeline: {e}", exc_info=True)
            video_response = f"_{timestamp_text}:_ \u274C Відео не вдалося проаналізувати: " + str(e)[:512] + "..."
            insignificant_frames = []
            clip_path = None

        battery = psutil.sensors_battery()
        if not battery.power_plugged and battery.percent <= 50:
            battery_time_left = time.strftime("%H:%M", time.gmtime(battery.secsleft))
            video_response += f"\n\U0001FAAB *{battery.percent}% ~{battery_time_left}*"

        # --- NEW: Acquire lock before interacting with shared state and Telegram API ---
        async with self.telegram_lock:
            # Make the global state variables accessible
            global no_motion_group_message_id, no_motion_grouped_videos

            # --- REFINED DECISION LOGIC ---
            is_significant_motion = clip_path is not None

            # Get relative path for callback data
            safe_video_folder = os.path.join(VIDEO_FOLDER, '')
            if file_path.startswith(safe_video_folder):
                callback_file = file_path[len(safe_video_folder):].replace(os.path.sep, '/')
            else:
                callback_file = file_basename

            if is_significant_motion:
                media_path = clip_path
                if not os.path.exists(media_path):
                    self.logger.warning(f"[{file_basename}] Highlight clip not found, using original video.")
                    media_path = file_path
                try:
                    # Button for significant motion video is still singular
                    keyboard = [[InlineKeyboardButton("Глянути", callback_data=callback_file)]]
                    reply_markup = InlineKeyboardMarkup(keyboard)
                    with open(media_path, 'rb') as animation_file:
                        sent_message = await self.app.bot.send_animation(
                            chat_id=CHAT_ID, animation=animation_file, caption=video_response,
                            reply_markup=reply_markup, parse_mode='Markdown'
                        )
                    self.logger.info(f"[{file_basename}] Animation sent successfully.")
                except telegram.error.BadRequest as bad_request_error:
                    self.logger.warning(f"[{file_basename}] BadRequest error: {bad_request_error}. Retrying with escaped Markdown.")
                    try:
                        with open(media_path, 'rb') as animation_file:
                            sent_message = await self.app.bot.send_animation(
                                chat_id=CHAT_ID,
                                animation=animation_file,
                                caption=escape_markdown(video_response, version=1),
                                reply_markup=reply_markup,
                                parse_mode='Markdown'
                            )
                        self.logger.info(f"[{file_basename}] Animation sent successfully after escaping Markdown.")
                    except Exception as retry_error:
                        self.logger.error(f"[{file_basename}] Failed to send animation after escaping Markdown: {retry_error}", exc_info=True)
                        # Fallback to sending a plain message with a button
                        self.logger.info(f"[{file_basename}] Sending plain message with button to Telegram...")
                        try:
                            sent_message = await self.app.bot.send_message(
                                chat_id=CHAT_ID,
                                text=video_response,
                                reply_markup=reply_markup,
                                parse_mode='Markdown'
                            )
                            self.logger.info(f"[{file_basename}] Plain message with button sent successfully.")
                        except telegram.error.BadRequest as bad_request_error_fallback:
                            self.logger.warning(f"[{file_basename}] BadRequest error on fallback: {bad_request_error_fallback}. Retrying with escaped Markdown.")
                            try:
                                sent_message = await self.app.bot.send_message(
                                    chat_id=CHAT_ID,
                                    text=escape_markdown(video_response, version=1),
                                    reply_markup=reply_markup,
                                    parse_mode='Markdown'
                                )
                                self.logger.info(f"[{file_basename}] Message sent successfully after escaping Markdown.")
                            except Exception as e_final_fallback:
                                self.logger.error(f"[{file_basename}] Failed to send message after escaping Markdown: {e_final_fallback}", exc_info=True)
                except Exception as e:
                    self.logger.error(f"[{file_basename}] Error sending animation: {e}", exc_info=True)
                    self.logger.info(f"[{file_basename}] Sending plain message with button to Telegram...")
                    try:
                        sent_message = await self.app.bot.send_message(
                            chat_id=CHAT_ID,
                            text=video_response,
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        self.logger.info(f"[{file_basename}] Plain message with button sent successfully.")
                    except telegram.error.BadRequest as bad_request_error:
                        self.logger.warning(f"[{file_basename}] BadRequest error: {bad_request_error}. Retrying with escaped Markdown.")
                        try:
                            sent_message = await self.app.bot.send_message(
                                chat_id=CHAT_ID,
                                text=escape_markdown(video_response, version=1),
                                reply_markup=reply_markup,
                                parse_mode='Markdown'
                            )
                            self.logger.info(f"[{file_basename}] Message sent successfully after escaping Markdown.")
                        except Exception as retry_error:
                            self.logger.error(f"[{file_basename}] Failed to send message after escaping Markdown: {retry_error}", exc_info=True)
                    except Exception as e_send:
                        self.logger.error(f"[{file_basename}] Failed to send plain message: {e_send}", exc_info=True)
                finally:
                    # Delete the generated clip after sending or if an error occurs
                    if media_path != file_path and os.path.exists(media_path):
                        await asyncio.sleep(10)  # Small delay to ensure file is not in use
                        max_wait = 120
                        waited = 0
                        was_locked = False
                        # Wait for the lock file to be released
                        while os.path.exists(media_path + ".lock") and waited < max_wait:
                            self.logger.info(f"[{file_basename}] Waiting for lock file on {media_path} to be released...")
                            was_locked = True
                            await asyncio.sleep(10)
                            waited += 10
                        if os.path.exists(media_path + ".lock"):
                            self.logger.warning(f"[{file_basename}] Lock file still exists after {max_wait} seconds. Proceeding to delete media file anyway.")
                        elif was_locked:
                            await asyncio.sleep(10)  # Small delay to ensure file is not in use
                        # Try deleting the file up to 3 times with 10s delay between attempts
                        for attempt in range(3):
                          try:
                            os.remove(media_path)
                            self.logger.info(f"[{file_basename}] Temporary media file deleted: {media_path}")
                            break
                          except Exception as e_del:
                            self.logger.warning(f"[{file_basename}] Failed to delete temporary media file {media_path} (attempt {attempt+1}/3): {e_del}")
                            if attempt < 2:
                              await asyncio.sleep(10)


            else: # --- This block now handles ALL non-significant videos ---
                video_info = {'text': video_response, 'callback': callback_file, 'timestamp': timestamp_text}

                # Condition to add to an existing group
                if no_motion_group_message_id and len(no_motion_grouped_videos) < 4:
                    self.logger.info(f"[{file_basename}] Adding to existing insignificant message group.")
                    no_motion_grouped_videos.append(video_info)
                    
                    # Build the updated message
                    full_text = "\n".join([v['text'] for v in no_motion_grouped_videos])
                    
                    # --- Create a single row of buttons ---
                    button_row = [InlineKeyboardButton(v['timestamp'], callback_data=v['callback']) for v in no_motion_grouped_videos]
                    reply_markup = InlineKeyboardMarkup([button_row]) # Note the double brackets [[...]]
                    
                    try:
                        await self.app.bot.edit_message_text(
                            chat_id=CHAT_ID,
                            message_id=no_motion_group_message_id,
                            text=full_text,
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        self.logger.info(f"[{file_basename}] Successfully edited message to extend group.")
                    except telegram.error.BadRequest as e:
                        if "message is not modified" in str(e).lower():
                            self.logger.info(f"[{file_basename}] Message was not modified, skipping edit.")
                        else:
                            self.logger.warning(f"[{file_basename}] Could not edit message: {e}. Retrying with escaped Markdown.")
                            try:
                                await self.app.bot.edit_message_text(
                                    chat_id=CHAT_ID,
                                    message_id=no_motion_group_message_id,
                                    text=escape_markdown(full_text, version=1),
                                    reply_markup=reply_markup,
                                    parse_mode='Markdown'
                                )
                                self.logger.info(f"[{file_basename}] Message edited successfully after escaping Markdown.")
                            except Exception as retry_error:
                                self.logger.error(f"[{file_basename}] Failed to edit message after escaping Markdown: {retry_error}", exc_info=True)
                                no_motion_group_message_id = None # Force a new message
                                no_motion_grouped_videos.clear()
                        
                elif no_motion_group_message_id is None or len(no_motion_grouped_videos) >= 4:
                    self.logger.info(f"[{file_basename}] Starting a new insignificant message group.")
                    no_motion_grouped_videos = [video_info]
                    
                    # Create the first button for the new message
                    button_row = [InlineKeyboardButton("Глянути", callback_data=v['callback']) for v in no_motion_grouped_videos]
                    reply_markup = InlineKeyboardMarkup([button_row])
                    
                    try:
                        sent_message = await self.app.bot.send_message(
                            chat_id=CHAT_ID,
                            text=video_info['text'],
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        no_motion_group_message_id = sent_message.message_id
                        self.logger.info(f"[{file_basename}] New group message sent. Message ID: {no_motion_group_message_id}")
                    except telegram.error.BadRequest as e:
                        self.logger.warning(f"[{file_basename}] Could not send new group message: {e}. Retrying with escaped Markdown.")
                        try:
                            sent_message = await self.app.bot.send_message(
                                chat_id=CHAT_ID,
                                text=escape_markdown(video_info['text'], version=1),
                                reply_markup=reply_markup,
                                parse_mode='Markdown'
                            )
                            no_motion_group_message_id = sent_message.message_id
                            self.logger.info(f"[{file_basename}] New group message sent successfully after escaping Markdown. Message ID: {no_motion_group_message_id}")
                        except Exception as retry_error:
                            self.logger.error(f"[{file_basename}] Failed to send new group message after escaping Markdown: {retry_error}", exc_info=True)
                            no_motion_group_message_id = None
                            no_motion_grouped_videos.clear()
                    except Exception as e_send:
                        self.logger.error(f"[{file_basename}] Failed to send new group message: {e_send}", exc_info=True)
                        no_motion_group_message_id = None
                        no_motion_grouped_videos.clear()
            
            if insignificant_frames:
                self.logger.info(f"[{file_basename}] Found {len(insignificant_frames)} insignificant motion frames to send.")
                media_group = []
                # Read all files into memory first to avoid issues with file handles
                frame_data = []
                for frame_path in insignificant_frames:
                    try:
                        with open(frame_path, 'rb') as photo_file:
                            frame_data.append(photo_file.read())
                    except Exception as e:
                        self.logger.error(f"[{file_basename}] Failed to read frame file {frame_path}: {e}")

                for data in frame_data:
                    media_group.append(InputMediaPhoto(media=data))

                if media_group:
                    try:
                        reply_to_id = None
                        if sent_message:
                            reply_to_id = sent_message.message_id
                        elif no_motion_group_message_id:
                            reply_to_id = no_motion_group_message_id

                        if reply_to_id:
                            await self.app.bot.send_media_group(
                                chat_id=CHAT_ID,
                                media=media_group,
                                reply_to_message_id=reply_to_id,
                                caption=f"_{timestamp_text}_ \U0001F4F8",
                                parse_mode='Markdown'
                            )
                            self.logger.info(f"[{file_basename}] Sent media group of {len(media_group)} insignificant frames as a reply.")
                        else:
                            self.logger.warning(f"[{file_basename}] No message ID to reply to. Sending media group without reply.")
                            await self.app.bot.send_media_group(chat_id=CHAT_ID, media=media_group, caption=f"_{timestamp_text}_ \U0001F4F8", parse_mode='Markdown')

                    except Exception as e:
                        self.logger.error(f"[{file_basename}] Failed to send media group: {e}", exc_info=True)
                
                # Clean up the temporary frame files
                for frame_path in insignificant_frames:
                    if os.path.exists(frame_path):
                        try:
                            os.remove(frame_path)
                            self.logger.info(f"[{file_basename}] Deleted temporary frame: {frame_path}")
                        except Exception as e:
                            self.logger.error(f"[{file_basename}] Failed to delete temporary frame {frame_path}: {e}")

            self.logger.info(f"[{file_basename}] Telegram interaction finished.")


    async def wait_for_file_stable(self, file_path, file_basename, wait_seconds=2, checks=2):
        """
        Waits until the file size hasn't changed for a certain period.
        This ensures the file is fully written before processing begins.

        Args:
            file_path (str): The path to the file to check.
            file_basename (str): The basename of the file for logging.
            wait_seconds (int): The interval between size checks.
            checks (int): The number of consecutive stable checks required.

        Raises:
            FileNotFoundError: If the file disappears during the check.
        """
        self.logger.debug(f"[{file_basename}] Checking file stability for: {file_path}")
        last_size = -1
        stable_checks = 0
        while stable_checks < checks:
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0: # Ensure size is not zero
                    stable_checks += 1
                    self.logger.debug(f"[{file_basename}] File size stable ({current_size} bytes), check {stable_checks}/{checks}.")
                else:
                    stable_checks = 0 # Reset if size changes or is zero
                    self.logger.debug(f"[{file_basename}] File size changed/zero ({last_size} -> {current_size} bytes). Resetting stability check.")
                last_size = current_size
            except FileNotFoundError:
                self.logger.warning(f"[{file_basename}] File not found during stability check: {file_path}")
                raise # Re-raise the error
            except Exception as e:
                 self.logger.error(f"[{file_basename}] Error checking file size: {e}", exc_info=True)
                 raise # Re-raise unexpected errors

            if stable_checks < checks:
                await asyncio.sleep(wait_seconds) # Wait before the next check

        self.logger.info(f"[{file_basename}] File considered stable at {last_size} bytes.")


# --- Callback Handler ---
async def button_callback(update, context):
    """
    Handles button presses from inline keyboards in Telegram messages.
    When a user clicks a "Глянути" button, this function sends the corresponding full video.

    Args:
        update (telegram.Update): The update object from the Telegram API.
        context (telegram.ext.ContextTypes.DEFAULT_TYPE): The context object.
    """
    query = update.callback_query
    await query.answer() # Acknowledge callback quickly

    # Normalize callback data to recreate the correct path
    callback_file_rel = query.data.replace('/', os.path.sep)
    file_path = os.path.join(VIDEO_FOLDER, callback_file_rel)
    file_basename = os.path.basename(file_path)

    logger.info(f"[{file_basename}] Button callback received for: {callback_file_rel}")

    if not os.path.exists(file_path):
        logger.error(f"[{file_basename}] Video file not found for callback: {file_path}")
        try:
            await query.edit_message_text(text=f"{query.message.text}\n\n_{file_basename[:6]}: Відео файл не знайдено._", parse_mode='Markdown')
        except Exception as edit_e:
            logger.error(f"[{file_basename}] Error editing message for not found file: {edit_e}", exc_info=True)
        return

    logger.info(f"[{file_basename}] Sending video from callback...")
    try:
        with open(file_path, 'rb') as video_file:
            await context.bot.send_video(
                chat_id=query.message.chat_id, video=video_file, parse_mode='Markdown',
                caption=f"Осьо відео _{file_basename[:6]}_", reply_to_message_id=query.message.message_id
            )
        logger.info(f"[{file_basename}] Video sent successfully from callback.")
    except FileNotFoundError:
         logger.error(f"[{file_basename}] Video file disappeared before sending from callback: {file_path}")
         try: await query.edit_message_text(text=f"{query.message.text}\n\n_{file_basename[:6]} Помилка: Відео файл зник._", parse_mode='Markdown')
         except Exception as edit_e:
            logger.warning(f"[{file_basename}] Failed to edit message after video disappeared: {edit_e}", exc_info=True)
    except Exception as e:
        logger.error(f"[{file_basename}] Failed to send video from callback: {e}", exc_info=True)
        pass


# Add the callback handler
application.add_handler(CallbackQueryHandler(button_callback))


# --- Main Execution and Shutdown Logic ---

async def run_telegram_bot(stop_event):
    """
    Initializes and runs the Telegram bot's polling mechanism.
    It listens for the `stop_event` to perform a graceful shutdown.

    Args:
        stop_event (asyncio.Event): An event that signals when the task should stop.
    """
    logger.info("Starting Telegram bot polling...")
    try:
        await application.initialize()
        logger.info("Telegram application initialized.")
        await application.start()
        logger.info("Telegram application started.")
        await application.updater.start_polling(poll_interval=1.0, timeout=20)
        logger.info("Telegram bot polling started.")
        await stop_event.wait() # Wait for the signal to stop
    except asyncio.CancelledError:
        logger.info("Telegram bot task cancelled.")
    except Exception as e:
        logger.error(f"Error in Telegram bot task: {e}", exc_info=True)
        stop_event.set() # Signal shutdown on error
    finally:
        logger.info("Stopping Telegram bot...")
        if application.updater and application.updater.running: # Check if running before stopping
             logger.info("Stopping updater polling...")
             await application.updater.stop()
        if application.running: # Check if running before stopping
            logger.info("Stopping application...")
            await application.stop()
        logger.info("Telegram bot stopped.")


async def run_file_watcher(stop_event):
    """
    Initializes and runs the watchdog observer to monitor the video folder.
    It listens for the `stop_event` to perform a graceful shutdown.

    Args:
        stop_event (asyncio.Event): An event that signals when the task should stop.
    """
    logger.info("Starting file watcher...")
    observer = None
    try:
        loop = asyncio.get_running_loop()
        event_handler = FileHandler(loop, application) # Pass configured app
        observer = Observer()
        # VIDEO_FOLDER existence already checked at startup

        observer.schedule(event_handler, path=VIDEO_FOLDER, recursive=True) # WATCH RECURSIVELY
        observer.start()
        logger.info(f"Watching for new files in: {VIDEO_FOLDER} (Recursive Mode)")

        while not stop_event.is_set(): # Check stop_event more frequently
             if not observer.is_alive():
                 logger.error("File watcher observer thread died unexpectedly.")
                 stop_event.set()
                 break
             await asyncio.sleep(1) # Check every second

    except asyncio.CancelledError:
        logger.info("File watcher task cancelled.")
    except Exception as e:
        logger.error(f"Error in File watcher task: {e}", exc_info=True)
        stop_event.set() # Signal shutdown on error
    finally:
        logger.info("Stopping file watcher...")
        if observer and observer.is_alive():
            observer.stop()
            try:
                # Give the observer thread some time to join
                observer.join(timeout=5.0)
                if observer.is_alive():
                     logger.warning("Observer thread did not stop cleanly after 5 seconds.")
            except Exception as e_join:
                 logger.error(f"Error joining observer thread: {e_join}", exc_info=True)

        logger.info("File watcher stopped.")

async def run_main_script_watcher(stop_event, script_to_watch):
    """
    Initializes and runs a watchdog observer to monitor the main script file for changes.
    If a change is detected, it sets the `stop_event` to trigger a graceful restart.

    Args:
        stop_event (asyncio.Event): The global stop event to set upon detecting a change.
        script_to_watch (str): The absolute path to the main script file.
    """
    logger.info(f"Starting self-watcher for script: {script_to_watch}")
    observer = None
    try:
        # Watch the directory containing the script, as watching a single file can be tricky
        # with how editors save (e.g., delete and rename)
        watch_dir = os.path.dirname(script_to_watch)
        event_handler = MainScriptChangeHandler(stop_event, script_to_watch)
        observer = Observer() # Using PollingObserver for consistency with your video watcher
        observer.schedule(event_handler, path=watch_dir, recursive=False) # Don't need recursive
        observer.start()
        logger.info(f"Self-watcher started for directory: {watch_dir} (monitoring {os.path.basename(script_to_watch)})")

        while not stop_event.is_set():
            if not observer.is_alive():
                logger.error("Self-watcher observer thread died unexpectedly.")
                stop_event.set() # Trigger shutdown if self-watcher fails
                break
            await asyncio.sleep(1)
        # If stop_event is set by MainScriptChangeHandler, this loop will also terminate.

    except asyncio.CancelledError:
        logger.info("Main script watcher task cancelled.")
    except Exception as e:
        logger.error(f"Error in Main script watcher task: {e}", exc_info=True)
        stop_event.set() # Signal shutdown on error
    finally:
        logger.info("Stopping main script watcher...")
        if observer and observer.is_alive():
            observer.stop()
            try:
                observer.join(timeout=2.0) # Shorter timeout for self-watcher
                if observer.is_alive():
                    logger.warning("Self-watcher observer thread did not stop cleanly after 2 seconds.")
            except Exception as e_join:
                logger.error(f"Error joining self-watcher observer thread: {e_join}", exc_info=True)
        logger.info("Main script watcher stopped.")

async def main():
    """
    The main entry point of the application.

    It sets up and runs the primary asynchronous tasks:
    - The Telegram bot poller.
    - The video folder file watcher.
    - The self-watcher for auto-restarts.

    It also handles graceful shutdown on Ctrl+C or task failure, and orchestrates
    the self-restart mechanism if the main script is modified.
    """
    stop_event = asyncio.Event()
    global RESTART_REQUESTED # Allow main to modify it if needed, though not strictly necessary here
    RESTART_REQUESTED = False # Ensure it's reset if main is somehow called again in same process (unlikely)

    # Use task names for better debugging if needed
    telegram_task = asyncio.create_task(run_telegram_bot(stop_event), name="TelegramBotTask")
    watcher_task = asyncio.create_task(run_file_watcher(stop_event), name="FileWatcherTask")
    main_script_monitor_task = asyncio.create_task(
        run_main_script_watcher(stop_event, MAIN_SCRIPT_PATH),
        name="MainScriptWatcherTask"
    )

    tasks = {telegram_task, watcher_task, main_script_monitor_task}
    logger.info("Application started. Press Ctrl+C to exit. Will auto-restart on main.py change.")

    # Monitor tasks
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for task in done:
            task_name = task.get_name()
            try:
                result = task.result() # Re-raises exception if task failed

                if RESTART_REQUESTED and stop_event.is_set(): # Primary check: Is a restart underway?
                    if task == main_script_monitor_task:
                        logger.info(f"Task '{task_name}' (script monitor) completed, as it initiated the restart sequence.")
                    else:
                        # Other tasks are completing because stop_event was set by the script monitor for restart
                        logger.info(f"Task '{task_name}' completed as part of the graceful shutdown for restart. Result: {result}")
                elif stop_event.is_set(): # If not a restart, but stop_event is set (e.g., Ctrl+C or error in another task)
                    logger.info(f"Task '{task_name}' completed during a general shutdown sequence. Result: {result}")
                else:
                    # No restart requested, and stop_event wasn't set before this task completed.
                    # This implies the task finished on its own, which is unexpected for long-running tasks.
                    logger.warning(f"Task '{task_name}' completed unexpectedly (no global stop signal was active). Result: {result}")
                    # Since this task finished unexpectedly, ensure the global stop_event is set to terminate others.
                    # This will be handled by the block after this loop if not already set by an exception.

            except asyncio.CancelledError:
                  logger.info(f"Task '{task_name}' was cancelled.")
                  # Cancellation is usually part of a stop sequence.
            except Exception as task_exc:
                logger.error(f"Task '{task_name}' failed with exception:", exc_info=task_exc)
                if RESTART_REQUESTED: # If a restart was pending, cancel it due to this error
                    logger.warning("Task failure detected, cancelling pending script restart.")
                    RESTART_REQUESTED = False
                # Ensure stop_event is set on failure, if not already.
                if not stop_event.is_set():
                    stop_event.set()

        # After processing tasks in 'done':
        # If stop_event is not set, it means a task in 'done' completed cleanly but unexpectedly
        # (and wasn't the restart monitor, and didn't cause an exception that set stop_event).
        # Or, it could mean 'done' is empty, which shouldn't happen with FIRST_COMPLETED unless tasks is empty.
        if not stop_event.is_set():
              # This implies a task from 'done' finished cleanly, and the 'else' for unexpected completion above was hit.
              # The warning for that specific task was logged. Now ensure overall shutdown for pending tasks.
              logger.warning("An unexpected task completion occurred. Initiating shutdown of remaining tasks...")
              stop_event.set()
        # If RESTART_REQUESTED is true, stop_event is already set by MainScriptChangeHandler.
        # If Ctrl+C happened, stop_event is set by the KeyboardInterrupt handler.
        # If a task errored/cancelled, the loop above set stop_event.

    except KeyboardInterrupt:
        logger.info("\nCtrl+C detected. Initiating graceful shutdown...")
        stop_event.set()
        RESTART_REQUESTED = False # Ctrl+C should not trigger a restart

    except asyncio.CancelledError:
        logger.info("Main task cancelled.")
        stop_event.set() # Ensure stop event is set if main is cancelled externally

    finally:
        logger.info("Shutdown sequence started.")
        if not stop_event.is_set():
            logger.warning("Shutdown sequence started but stop_event was not set. Setting now.")
            stop_event.set() # Ensure stop_event is set

        logger.info("Waiting for tasks to finish...")
        # Wait for all tasks to complete shutdown routines
        # Using gather allows catching exceptions from tasks during shutdown as well
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for i, result in enumerate(results):
             task = list(tasks)[i] # Order might not be guaranteed, but helps identify
             task_name = task.get_name()
             if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
                 logger.error(f"Task '{task_name}' raised an exception during shutdown: {result}", exc_info=result)
             elif isinstance(result, asyncio.CancelledError):
                 logger.info(f"Task '{task_name}' was cancelled during shutdown.")
             else:
                 logger.info(f"Task '{task_name}' finished shutdown cleanly.")

        logger.info("All application tasks have finished.")

        if RESTART_REQUESTED:
            logger.info("Fast shutdown for restart: Not waiting for current analysis to finish.")
            # Shutdown immediately without waiting for the worker.
            motion_executor.shutdown(wait=False, cancel_futures=True)
            io_executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Executors issued fast shutdown command.")

            logger.info("RESTART_REQUESTED is True. Executing self-restart...")
            # Flush standard streams before exec, as they might be inherited
            sys.stdout.flush()
            sys.stderr.flush()
            # Replace the current process with a new one
            # sys.executable is the path to the Python interpreter
            # sys.argv are the original command-line arguments
            try:
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e_exec:
                # This part will only be reached if os.execv fails, which is rare
                logger.critical(f"FATAL: os.execv failed during restart attempt: {e_exec}", exc_info=True)
                # At this point, the script cannot restart itself and will exit.
        else:
            logger.info("Graceful shutdown: Allowing current analysis to finish...")
            # For a normal shutdown (e.g., Ctrl+C), we wait for the current task to complete.
            motion_executor.shutdown(wait=True, cancel_futures=False)
            io_executor.shutdown(wait=True, cancel_futures=False)
            logger.info("Main application finished cleanly (no restart requested).")


# Run the main function
if __name__ == "__main__":
    try:
        # Basic check before starting async loop
        if not VIDEO_FOLDER or not os.path.isdir(VIDEO_FOLDER):
             print(f"ERROR: VIDEO_FOLDER '{VIDEO_FOLDER}' is not set or not a valid directory. Cannot start.")
        else:
             asyncio.run(main())
    except KeyboardInterrupt:
        # This catch is mainly to prevent the final asyncio traceback on Ctrl+C
        # The actual handling is inside main()
        logger.info("\nExiting application due to KeyboardInterrupt in __main__.")
    except Exception as main_e:
        # Catch any unexpected errors during asyncio.run(main()) itself
        logger.critical(f"Critical error during application startup or main loop: {main_e}", exc_info=True)
