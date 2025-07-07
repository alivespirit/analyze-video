import os
import datetime
import asyncio
import concurrent.futures
import logging
import re
import sys
import time
import json
import cv2
import numpy as np

from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton
import telegram.error
from telegram.ext import Application, CallbackQueryHandler
from telegram.helpers import escape_markdown
from google import genai
from google.genai import types

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

from logging.handlers import TimedRotatingFileHandler

# Importing VideoFileClip from moviepy for MP4 extraction
from moviepy import ImageSequenceClip

import threading

GEMINI_CONCURRENCY_LIMIT = 1  # Only one Gemini call at a time (adjust if safe)
gemini_semaphore = threading.Semaphore(GEMINI_CONCURRENCY_LIMIT)

# --- State for Grouping "No Motion" Messages ---
# These are safe to use as globals because the executor has max_workers=1,
# ensuring sequential processing and preventing race conditions.
no_motion_group_message_id = None
no_motion_grouped_videos = []

RESTART_REQUESTED = False
MAIN_SCRIPT_PATH = os.path.abspath(__file__)

class MainScriptChangeHandler(FileSystemEventHandler):
    def __init__(self, stop_event_ref, script_path_ref):
        self.stop_event = stop_event_ref
        self.script_path = script_path_ref
        self.triggered_restart = False # Ensure we only trigger once

    def on_modified(self, event):
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

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set root logger level
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# ---------------------

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("moviepy").setLevel(logging.WARNING) # Keep moviepy logs quiet

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
USERNAME = os.getenv("TELEGRAM_NOTIFY_USERNAME")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER")

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
MIN_SOLIDITY = 0.8              # A person is a very solid shape. 1.0 is a perfect rectangle.
ROI_CONFIG_FILE = os.path.join(SCRIPT_DIR, "roi.json")
PADDING_SECONDS = 1.0
# ### WARM-UP CONFIGURATION ###
# Number of initial frames to ignore while the background model stabilizes.
# A value of 25-50 is usually good for a 25fps video (1-2 seconds).
WARMUP_FRAMES = 15
# ### CONFIGURATION FOR EVENT GROUPING ###
# If motion stops for more than this many seconds, start a new clip segment.
MAX_EVENT_GAP_SECONDS = 3.0
# ### DURATION FILTER CONFIGURATION ###
# Discard any motion event that lasts for less than this many seconds.
MIN_EVENT_DURATION_SECONDS = 2.0
# ### PERFORMANCE CONFIGURATION ###
CROP_PADDING = 30  # Pixels to add around the ROI bounding box for safety
# ### SANITY CHECK CONFIGURATION ###
# If a bounding box is larger than this percentage of the total analysis area,
# discard it as it's likely a noise-polluted first frame.
MAX_BOX_AREA_PERCENT = 0.80 

# --- Check Environment Variables ---
if not all([GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID, USERNAME, VIDEO_FOLDER]):
    logger.critical("ERROR: One or more essential environment variables are missing. Exiting.")
    exit(1) # Exit if critical env vars are missing
if not os.path.isdir(VIDEO_FOLDER):
    logger.critical(f"ERROR: VIDEO_FOLDER '{VIDEO_FOLDER}' does not exist or is not a directory. Exiting.")
    exit(1)
# -----------------------------------

try:
    client = genai.Client(api_key=GEMINI_API_KEY)
    logger.info("Gemini configured successfully.")
except Exception as e:
    logger.critical(f"Failed to configure Gemini: {e}", exc_info=True) # Use exc_info for traceback
    exit(1)


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
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"ThreadPoolExecutor initialized with a single worker (max_workers=1) to ensure sequential video processing.")
except Exception as e:
     logger.critical(f"Failed to initialize ThreadPoolExecutor: {e}", exc_info=True)
     exit(1)


# --- ROI Loading Function ---
def load_roi(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            points = json.load(f)
        return np.array(points, dtype=np.int32)
    return None

def is_likely_person_or_object(contour):
    """
    Checks if a contour has the shape properties of a person or solid object.
    Returns True if it passes the checks, False otherwise.
    """
    # 1. Check Area (basic filter)
    area = cv2.contourArea(contour)
    if area < MIN_CONTOUR_AREA:
        return False

    # 2. Check Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = float(area) / hull_area if hull_area > 0 else 0
    if solidity < MIN_SOLIDITY:
        return False
    
    # If all checks pass, it's likely a person/solid object
    return True

def detect_motion(input_video_path, output_dir):
    file_basename = os.path.basename(input_video_path)
    roi_poly_points = load_roi(ROI_CONFIG_FILE)
    if roi_poly_points is None:
        logger.error(f"[{file_basename}] ROI config file not found.")
        return None

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        logger.error(f"[{file_basename}] Could not open video file {input_video_path}")
        return None
    
    start_time = time.time()

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # ### DEFINE THE CROP BOX BASED ON THE ROI ###
    # 2. Calculate the bounding box of the ROI polygon
    x_coords = roi_poly_points[:, 0]
    y_coords = roi_poly_points[:, 1]
    # Add padding to ensure we don't cut off anything at the edges
    crop_x1 = max(0, np.min(x_coords) - CROP_PADDING)
    crop_y1 = max(0, np.min(y_coords) - CROP_PADDING)
    crop_x2 = min(orig_w, np.max(x_coords) + CROP_PADDING)
    crop_y2 = min(orig_h, np.max(y_coords) + CROP_PADDING)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    logger.info(f"[{file_basename}] Original frame: {orig_w}x{orig_h}. Analyzing cropped region: {crop_w}x{crop_h} at ({crop_x1},{crop_y1}).")

    # ### MODIFIED: ALL SUBSEQUENT SETUP IS RELATIVE TO THE CROP ###
    # 3. Translate the global ROI points to be local to the crop box
    local_roi_points = roi_poly_points.copy()
    local_roi_points[:, 0] -= crop_x1
    local_roi_points[:, 1] -= crop_y1
    
    # The ROI points are now simply the local points, not scaled.
    analysis_roi_points = local_roi_points.astype(np.int32)
    
    backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=800, detectShadows=True)
    # The mask is now the size of the CROP, not a downscaled version.
    roi_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(roi_mask, [analysis_roi_points], 255)
    
    motion_events = []
    for frame_index in range(WARMUP_FRAMES, total_frames):
        ret, frame = cap.read()
        if not ret: break

        # CROP THE FRAME FIRST!
        cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        # All subsequent analysis is now super fast
        roi_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=roi_mask)
        # ... (the rest of the loop is the same: fg_mask, morphology, findContours) ...
        fg_mask = backSub.apply(roi_frame)
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        person_like_contours = [cnt for cnt in contours if is_likely_person_or_object(cnt)]
        if person_like_contours:
            motion_events.append((frame_index, person_like_contours))

    if not motion_events:
        logger.info(f"[{file_basename}] No significant motion found.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        return "No motion"

    # ### Sanity check added to the creation of union_rects ###
    union_rects = {}
    discarded_boxes = []
    motion_event_dict = dict(motion_events)
    # The total area of our analysis region (the cropped image)
    total_analysis_area = crop_w * crop_h
    max_allowed_area = total_analysis_area * MAX_BOX_AREA_PERCENT

    for frame_idx, contours in motion_event_dict.items():
        min_x, min_y, max_x, max_y = float('inf'), float('inf'), float('-inf'), float('-inf')
        for cnt in contours:
            (x, y, w, h) = cv2.boundingRect(cnt)
            min_x, min_y = min(min_x, x), min(min_y, y)
            max_x, max_y = max(max_x, x + w), max(max_y, y + h)
        
        if min_x != float('inf'):
            box_w = max_x - min_x
            box_h = max_y - min_y
            box_area = box_w * box_h
            
            # THE SANITY CHECK:
            if box_area < max_allowed_area:
                union_rects[frame_idx] = (min_x, min_y, box_w, box_h)
            else:
                discarded_boxes.append(frame_idx)
                logger.debug(f"[{file_basename}] Frame {frame_idx}: Discarding giant bounding box (area {box_area} > max {max_allowed_area}).")

    if discarded_boxes:
        logger.info(f"[{file_basename}] Discarded {len(discarded_boxes)} bounding boxes as noise due to excessive size. Discarded frames: {discarded_boxes}")

    # If the sanity check removed all detections, it was just noise.
    if not union_rects:
        logger.info(f"[{file_basename}] All detected motion boxes were too large and discarded as noise.")
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        cap.release()
        return "No motion"

    # Now, all subsequent logic operates on the CLEANED union_rects data
    motion_frame_indices = sorted(union_rects.keys())
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

    logger.info(f"[{file_basename}] Found {len(sub_clips)} raw motion event(s). Filtering by duration...")
    significant_sub_clips = []
    for start_frame, end_frame in sub_clips:
        duration_frames = end_frame - start_frame
        duration_seconds = duration_frames / fps
        if duration_seconds >= MIN_EVENT_DURATION_SECONDS:
            logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is SIGNIFICANT. Keeping.")
            significant_sub_clips.append((start_frame, end_frame))
        else:
            logger.info(f"[{file_basename}]   - Event lasting {duration_seconds:.2f}s is too short. Discarding as noise/shadow.")
    
    # If all events were filtered out, we're done.
    if not significant_sub_clips:
        logger.info(f"[{file_basename}] No significant long-duration motion found. Discarding video.")
        cap.release()
        elapsed_time = time.time() - start_time
        logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
        return "No significant motion"

    # ##############################################################################
    # ### INTERPOLATE RECTANGLES  ###
    # ##############################################################################
    logger.info(f"[{file_basename}] Interpolating bounding boxes for smooth tracking...")

    # Create the final, complete dictionary of rectangles by filling gaps.
    interpolated_rects = {}
    for start_frame, end_frame in significant_sub_clips:
        known_frames_in_clip = sorted([f for f in union_rects.keys() if start_frame <= f <= end_frame])
        if not known_frames_in_clip: continue

        for i in range(len(known_frames_in_clip) - 1):
            start_anchor_frame, end_anchor_frame = known_frames_in_clip[i], known_frames_in_clip[i+1]
            start_box, end_box = union_rects[start_anchor_frame], union_rects[end_anchor_frame]
            total_gap_frames = float(end_anchor_frame - start_anchor_frame)
            
            interpolated_rects[start_anchor_frame] = start_box # Copy the known start box

            if total_gap_frames > 0:
                for j in range(1, int(total_gap_frames)):
                    current_frame_in_gap = start_anchor_frame + j
                    progress = j / total_gap_frames
                    interp_x = int(start_box[0] + (end_box[0] - start_box[0]) * progress)
                    interp_y = int(start_box[1] + (end_box[1] - start_box[1]) * progress)
                    interp_w = int(start_box[2] + (end_box[2] - start_box[2]) * progress)
                    interp_h = int(start_box[3] + (end_box[3] - start_box[3]) * progress)
                    interpolated_rects[current_frame_in_gap] = (interp_x, interp_y, interp_w, interp_h)
        
        if known_frames_in_clip: # Ensure the very last known frame is included
            interpolated_rects[known_frames_in_clip[-1]] = union_rects[known_frames_in_clip[-1]]


    # Final frame assembly now uses the clean `interpolated_rects` data ###
    logger.info(f"[{file_basename}] Assembling highlight reel with {len(interpolated_rects)} smooth boxes.")
    all_clip_frames = []

    for start_frame, end_frame in significant_sub_clips:
        padded_start = max(0, start_frame - int(PADDING_SECONDS * fps))
        padded_end = min(total_frames, end_frame + int(PADDING_SECONDS * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, padded_start)
        
        for i in range(padded_start, padded_end):
            ret, frame = cap.read()
            if not ret: break

            if i in interpolated_rects:
                # The drawing logic is now simpler and cleaner.
                (x, y, w, h) = interpolated_rects[i]
                
                orig_x1 = int(x + crop_x1)
                orig_y1 = int(y + crop_y1)
                orig_x2 = int(x + w + crop_x1)
                orig_y2 = int(y + h + crop_y1)
                
                cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_clip_frames.append(rgb_frame)

    cap.release()

    if not all_clip_frames:
        logger.error(f"[{file_basename}] No frames collected for the clip.")
        return None
        
    final_clip = ImageSequenceClip(all_clip_frames, fps=fps)
    output_filename = os.path.join(output_dir, file_basename)
    logger.info(f"[{file_basename}] Writing final highlight clip to {output_filename}...")
    final_clip.write_videofile(
        output_filename, codec='libx264', audio=False, bitrate='2000k',
        preset='medium', threads=4, logger=None
    )
    
    file_size = os.path.getsize(output_filename) / (1024 * 1024)
    logger.info(f"[{file_basename}] Successfully created clip: {output_filename}. Size: {file_size:.2f} MB")
    elapsed_time = time.time() - start_time
    logger.info(f"[{file_basename}] Motion detection took {elapsed_time:.2f} seconds.")
    return output_filename

# --- analyze_video function (remains synchronous, uses original models) ---
def analyze_video(video_path):
    """Extract insights from the video using Gemini. Runs in executor."""
    file_basename = os.path.basename(video_path)
    timestamp = f"_{file_basename[:6]}:_ "
    video_bytes = None
    use_files_api = False
    now = datetime.datetime.now()

    detected_motion = detect_motion(video_path, TEMP_DIR)
    if detected_motion == "No motion":
        logger.info(f"[{file_basename}] Skipping Gemini analysis.")
        return timestamp + "Нема шо дивитись."
    elif detected_motion == "No significant motion":
        logger.info(f"[{file_basename}] Analyzing full video.")
        video_bytes = open(video_path, 'rb').read()
    elif detected_motion is None:
        logger.warning(f"[{file_basename}] Error during motion detection. Analyzing full video.")
        video_bytes = open(video_path, 'rb').read()
    else:
        logger.info(f"[{file_basename}] Running Gemini analysis for {detected_motion}")
        timestamp = f"_{file_basename[:6]}:_ *Отакої!* "
        use_files_api = False # Switched to inline_data mode for better performance
        if use_files_api:
            video_bytes = client.files.upload(file=detected_motion)
            # Wait up to 2 minutes (120 seconds) for video processing
            max_wait_seconds = 120
            wait_interval = 10
            waited = 0
            while video_bytes.state == "PROCESSING":
                if waited >= max_wait_seconds:
                    logger.error(f"[{file_basename}] Video processing timed out after {max_wait_seconds} seconds.")
                    return timestamp + "Відео не вдалося обробити (timeout)."
                logger.info(f"[{file_basename}] Waiting for video to be processed ({waited}/{max_wait_seconds}s).")
                time.sleep(wait_interval)
                waited += wait_interval
                video_bytes = client.files.get(name=video_bytes.name)

            if video_bytes.state == "FAILED":
                logger.error(f"[{file_basename}] Video processing failed: {video_bytes.error_message}")
                return timestamp + "Відео не вдалося обробити."
        else:
            try:
                video_bytes = open(detected_motion, 'rb').read()
            except Exception as e:
                logger.error(f"[{file_basename}] Error reading video file: {e}")
                return timestamp + "Відео не вдалося прочитати."

    if os.path.exists(os.path.join(SCRIPT_DIR, "enable_pro")) and (9 <= now.hour <= 13):
        # If it's between 9:00 and 13:59, use the Pro model
        model_main = 'gemini-2.5-pro'
        model_fallback = 'gemini-2.5-flash'
        model_fallback_text = '_[2.5F]_ '
    else:
        # Outside of that time, use the Flash models
        model_main = 'gemini-2.5-flash'
        model_fallback = 'gemini-2.5-flash-lite-preview-06-17'
        model_fallback_text = '_[2.5FL]_ '

    model_fallback_2_0 = 'gemini-2.0-flash'
    model_fallback_2_0_text = '_[2.0]_ '

    sampling_rate = 5  # sampling rate in FPS, valid only for inline_data
    max_retries = 3

    try:
        prompt_file_path = os.path.join(os.path.dirname(__file__), "prompt.txt")
        try:
            with open(prompt_file_path, "r", encoding="utf-8") as prompt_file:
                prompt = prompt_file.read().strip()
            logger.debug(f"[{file_basename}] Prompt loaded successfully from {prompt_file_path}.")
        except FileNotFoundError:
            logger.error(f"[{file_basename}] Prompt file not found: {prompt_file_path}")
            return timestamp + "Prompt file not found."
        except Exception as e:
            logger.error(f"[{file_basename}] Error reading prompt file: {e}", exc_info=True)
            return timestamp + "Error reading prompt file."

        if use_files_api:
            contents = [video_bytes, prompt]
        else:
            contents = types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(
                            data=video_bytes,
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
                with gemini_semaphore:
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
                analysis_result = response.text
                break
            except Exception as e_main:
                try:
                    with gemini_semaphore:
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
                    analysis_result = model_fallback_text + response.text
                    break
                except Exception as e_fallback:
                    logger.warning(f"[{file_basename}] {model_fallback} also failed: {e_fallback}")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s
                        logger.warning(f"[{file_basename}] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        try:
                            logger.info(f"[{file_basename}] Attempting {model_fallback_2_0} as final fallback...")
                            response = client.models.generate_content(
                                          model=model_fallback_2_0,
                                          contents=contents,
                                          config=types.GenerateContentConfig(
                                              automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                                  disable=True
                                              )
                                          )
                                      )
                            logger.info(f"[{file_basename}] {model_fallback_2_0} response received.")
                            analysis_result = model_fallback_2_0_text + response.text
                            break
                        except Exception as e_fallback_2_0:
                            logger.error(f"[{file_basename}] {model_fallback_2_0} failed as well: {e_fallback_2_0}")
                            logger.error(f"[{file_basename}] Giving up after retries.")
                            raise # Re-raise the exception to handle it in the outer scope

        logger.info(f"[{file_basename}] Response: {analysis_result}")

        if analysis_result == None or analysis_result.strip() == "":
            logger.error(f"[{file_basename}] Analysis result is empty or None.")
            analysis_result = "Empty analysis response. Reason: " + response.candidates[0].finish_reason.name
        else:
            analysis_result = (analysis_result[:512] + '...') if len(analysis_result) > 1023 else analysis_result

        # Notify username if needed
        if ("Отакої!" in analysis_result or "Отакої!" in timestamp) and (9 <= now.hour <= 13):
            additional_text += "\n" + USERNAME

        if use_files_api:
            try:
                client.files.delete(name=video_bytes.name)  # Clean up uploaded file
            except Exception as e_delete:
                logger.warning(f"[{file_basename}] Failed to delete uploaded file: {e_delete}", exc_info=False)

        return timestamp + analysis_result + additional_text

    except Exception as e_analysis:
        logger.error(f"[{file_basename}] Video analysis failed: {e_analysis}", exc_info=True)
        return timestamp + "Video analysis failed."

# --- FileHandler (uses executor) ---
class FileHandler(FileSystemEventHandler):
    def __init__(self, loop, app):
        self.loop = loop
        self.app = app
        self.logger = logging.getLogger(__name__)

    def on_created(self, event):
        if event.is_directory: return
        if not event.src_path.endswith('.mp4'): return

        coro = self.handle_event(event)
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            self.logger.warning(f"Event loop not running when trying to schedule handler for {event.src_path}")

    async def handle_event(self, event):
        # This function is replaced with the new logic
        file_path = event.src_path
        file_basename = os.path.basename(file_path)
        self.logger.info(f"[{file_basename}] New file detected: {file_path}")

        try:
            await self.wait_for_file_stable(file_path, file_basename)
        except FileNotFoundError:
            self.logger.warning(f"[{file_basename}] File disappeared before analysis could start.")
            return
        except Exception as e_wait:
            self.logger.error(f"[{file_basename}] Error waiting for file stability: {e_wait}", exc_info=True)
            return

        try:
            current_loop = asyncio.get_running_loop()
            video_response = await current_loop.run_in_executor(executor, analyze_video, file_path)
            self.logger.info(f"[{file_basename}] Analysis complete.")
        except Exception as e:
            self.logger.error(f"[{file_basename}] Error running analyze_video in executor: {e}", exc_info=True)
            video_response = f"_{file_basename[:6]}:_ Failed to analyze video."

        # Make the global state variables accessible
        global no_motion_group_message_id, no_motion_grouped_videos

        # --- REFINED DECISION LOGIC ---
        is_significant_motion = "Отакої!" in video_response

        # Get relative path for callback data
        safe_video_folder = os.path.join(VIDEO_FOLDER, '')
        if file_path.startswith(safe_video_folder):
            callback_file = file_path[len(safe_video_folder):].replace(os.path.sep, '/')
        else:
            callback_file = file_basename
        
        # Extract just the timestamp part for button text
        timestamp_text = file_basename[:6]


        if is_significant_motion:
            # A significant motion video ALWAYS resets the group
            self.logger.info(f"[{file_basename}] Significant motion detected. Resetting grouped message.")
            no_motion_group_message_id = None
            no_motion_grouped_videos.clear()

            # --- This is your existing, unchanged logic for sending animations ---
            media_path = os.path.join(TEMP_DIR, file_basename)
            if not os.path.exists(media_path):
                self.logger.warning(f"[{file_basename}] Highlight clip not found, using original video.")
                media_path = file_path
            try:
                # Button for significant motion video is still singular
                keyboard = [[InlineKeyboardButton("Глянути", callback_data=callback_file)]]
                reply_markup = InlineKeyboardMarkup(keyboard)
                with open(media_path, 'rb') as animation_file:
                    await self.app.bot.send_animation(
                        chat_id=CHAT_ID, animation=animation_file, caption=video_response,
                        reply_markup=reply_markup, parse_mode='Markdown'
                    )
                self.logger.info(f"[{file_basename}] Animation sent successfully.")
            except Exception as e:
                self.logger.error(f"[{file_basename}] Error sending animation: {e}", exc_info=True)
                self.logger.info(f"[{file_basename}] Sending plain message with button to Telegram...")
                try:
                    await self.app.bot.send_message(
                        chat_id=CHAT_ID,
                        text=video_response,
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                    self.logger.info(f"[{file_basename}] Plain message with button sent successfully.")
                except telegram.error.BadRequest as bad_request_error:
                    self.logger.warning(f"[{file_basename}] BadRequest error: {bad_request_error}. Retrying with escaped Markdown.")
                    try:
                        await self.app.bot.send_message(
                            chat_id=CHAT_ID,
                            text=escape_markdown(video_response, version=1),
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                        self.logger.info(f"[{file_basename}] Message sent successfully after escaping Markdown.")
                    except Exception as retry_error:
                        self.logger.error(f"[{file_basename}] Failed to send message after escaping Markdown: {retry_error}", exc_info=True)
                except Exception as e:
                    self.logger.error(f"[{file_basename}] Error sending message: {e}", exc_info=True)
            finally:
                # Delete the generated clip after sending or if an error occurs
                if media_path != file_path and os.path.exists(media_path):
                    max_wait = 120
                    waited = 0
                    while os.path.exists(media_path + ".lock") and waited < max_wait:
                        self.logger.info(f"[{file_basename}] Waiting for lock file to be released before deleting: {media_path}.lock (waited {waited}s)")
                        await asyncio.sleep(10)
                        waited += 10
                    if os.path.exists(media_path + ".lock"):
                        self.logger.warning(f"[{file_basename}] Lock file still exists after {max_wait} seconds. Proceeding to delete media file anyway.")
                    os.remove(media_path)
                    self.logger.info(f"[{file_basename}] Temporary media file deleted: {media_path}")

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
                    self.logger.warning(f"[{file_basename}] Could not edit message (may be unchanged or deleted): {e}. Starting a new group.")
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
                except Exception as e:
                    self.logger.error(f"[{file_basename}] Failed to send new group message: {e}", exc_info=True)
                    no_motion_group_message_id = None
                    no_motion_grouped_videos.clear()
        
        self.logger.info(f"[{file_basename}] Telegram interaction finished.")


    async def wait_for_file_stable(self, file_path, file_basename, wait_seconds=2, checks=2):
        """Waits until the file size hasn't changed for a certain period."""
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
    """Run the Telegram bot until stop_event is set."""
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
    """Run the file-watching logic until stop_event is set."""
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
    """Run a watcher for the main script file."""
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
    """Run bot and watcher, handle graceful shutdown and potential restart."""
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
        # If a task errored/cancelled, the loop above likely set stop_event.

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
            executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Executor issued fast shutdown command.")

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
            executor.shutdown(wait=True, cancel_futures=False)
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