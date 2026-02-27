import os
import asyncio
import concurrent.futures
import logging
import sys
import time
import json
import psutil
import threading
import shutil
import re
from types import SimpleNamespace
from datetime import datetime, timedelta

from dotenv import load_dotenv
from telegram.ext import Application, CallbackQueryHandler, MessageReactionHandler

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

from logging.handlers import TimedRotatingFileHandler


RESTART_REQUESTED = False
MAIN_SCRIPT_PATH = os.path.abspath(__file__)

# Load environment variables early, before reading any config from os.getenv.
# Important: override=True ensures that changes to .env take effect across self-restarts
# (os.execv inherits the parent environment). If you need a different env file, set DOTENV_PATH.
DOTENV_LOADED_FROM = None
try:
    dotenv_path = os.getenv("DOTENV_PATH")
    if not dotenv_path:
        dotenv_path = os.path.join(os.path.dirname(MAIN_SCRIPT_PATH), ".env")
    if dotenv_path and os.path.exists(dotenv_path):
        DOTENV_LOADED_FROM = dotenv_path
        load_dotenv(dotenv_path=dotenv_path, override=True)
    else:
        # Fallback: default search behavior (current working directory, etc.)
        load_dotenv(override=True)
except Exception:
    # Never fail startup because of dotenv issues.
    pass

class MainScriptChangeHandler(FileSystemEventHandler):
    def __init__(self, stop_event_ref, watch_dir):
        self.stop_event = stop_event_ref
        self.watch_dir = os.path.abspath(watch_dir)
        self.triggered_restart = False # Ensure we only trigger once

    def on_modified(self, event):
        """
        Handles the file modification event from watchdog.
        If a top-level Python file (*.py) in SCRIPT_DIR is modified, trigger graceful restart.
        """
        if self.triggered_restart:
            return

        changed_path = os.path.abspath(event.src_path)
        # Only react to .py files directly under SCRIPT_DIR (non-recursive)
        if os.path.dirname(changed_path) == self.watch_dir and changed_path.endswith('.py'):
            logger.info(f"Detected change in {changed_path}. Initiating graceful restart...")
            global RESTART_REQUESTED
            RESTART_REQUESTED = True
            self.stop_event.set() # Signal all other components to stop
            self.triggered_restart = True # Prevent further triggers from this handler instance

class LogDashboardChangeHandler(FileSystemEventHandler):
    def __init__(self, runner, exts=(".py", ".html", ".css")):
        self.runner = runner
        self.exts = exts
        self._last_restart = 0

    def on_modified(self, event):
        if event.is_directory:
            return
        if not any(event.src_path.endswith(ext) for ext in self.exts):
            return
        # Debounce rapid successive changes
        now = time.time()
        if now - self._last_restart < 0.5:
            return
        self._last_restart = now
        logger.info(f"Log dashboard change detected in {event.src_path}. Restarting dashboard...")
        try:
            self.runner.restart()
            logger.info("Log dashboard restarted.")
        except Exception as e:
            logger.error(f"Failed to restart log dashboard: {e}", exc_info=True)

class DashboardRunner:
    def __init__(self, app, host: str, port: int, log_level: str = "info"):
        self.app = app
        self.host = host
        self.port = port
        self.log_level = log_level
        self.server = None
        self.thread = None

    def start(self):
        import uvicorn
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level=self.log_level)
        self.server = uvicorn.Server(config)

        def _run_dashboard():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.server.run()

        self.thread = threading.Thread(target=_run_dashboard, name="LogDashboard", daemon=True)
        self.thread.start()
        logger.info(f"Log dashboard started at http://{self.host}:{self.port}")

    def stop(self):
        if self.server is not None:
            try:
                self.server.should_exit = True
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            try:
                self.thread.join(timeout=3.0)
            except Exception:
                pass
        self.server = None
        self.thread = None

    def restart(self):
        self.stop()
        self.start()

LOG_PATH = os.getenv("LOG_PATH", default="")
ENABLE_LOG_DASHBOARD = os.getenv("ENABLE_LOG_DASHBOARD", "false").lower() == "true"
LOG_DASHBOARD_PORT = int(os.getenv("LOG_DASHBOARD_PORT", "8000"))
LOG_DASHBOARD_HOST = os.getenv("LOG_DASHBOARD_HOST", "0.0.0.0")
RESTART_RECOVERY_WINDOW_SECONDS = int(os.getenv("RESTART_RECOVERY_WINDOW_SECONDS", "180"))
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "8"))
RETENTION_CLEANUP_TIME = os.getenv("RETENTION_CLEANUP_TIME", "00:10")  # HH:MM
TEMP_RETENTION_DAYS = 30

class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def rotation_filename(self, default_name):
        """
        Produce a custom rotated filename without a post-rename step.

        Default TimedRotatingFileHandler builds `default_name` like:
        <dir>/<basename>.YYYY-MM-DD (for when="midnight")

        We transform it to:
        <dir>/<basename_without_ext>_YYYY-MM-DD.log
        """
        dir_name, base_name = os.path.split(self.baseFilename)
        # Extract the date suffix from default_name (everything after "<base_name>.")
        _, default_base = os.path.split(default_name)
        if default_base.startswith(base_name + "."):
            suffix = default_base[len(base_name) + 1:]
        else:
            # Fallback: use the default base as-is (unlikely, but safe)
            suffix = default_base

        # Remove trailing .log only if present
        base_no_ext = base_name[:-4] if base_name.endswith(".log") else base_name
        return os.path.join(dir_name, f"{base_no_ext}_{suffix}.log")

    def getFilesToDelete(self):
        """
        Return the list of old rotated files to delete, honoring backupCount.

        Supports both patterns to ensure cleanup works even if older runs used
        the default naming or the custom naming:
          - <basename>.YYYY-MM-DD
          - <basename_without_ext>_YYYY-MM-DD.log
        """
        if self.backupCount <= 0:
            return []

        dir_name, base_name = os.path.split(self.baseFilename)
        base_no_ext = base_name[:-4] if base_name.endswith(".log") else base_name

        candidates = []
        try:
            for fname in os.listdir(dir_name or "."):
                # Pattern 1: default rotation `<basename>.YYYY-MM-DD`
                if fname.startswith(base_name + "."):
                    suffix = fname[len(base_name) + 1:]
                    if hasattr(self, "extMatch") and self.extMatch and self.extMatch.match(suffix):
                        candidates.append(os.path.join(dir_name, fname))
                        continue

                # Pattern 2: custom rotation `<basename_without_ext>_YYYY-MM-DD.log`
                if fname.startswith(base_no_ext + "_") and fname.endswith(".log"):
                    suffix = fname[len(base_no_ext) + 1:-4]
                    if hasattr(self, "extMatch") and self.extMatch and self.extMatch.match(suffix):
                        candidates.append(os.path.join(dir_name, fname))
        except FileNotFoundError:
            return []

        # Sort by modification time (oldest first) for robust cleanup
        try:
            candidates.sort(key=lambda p: os.path.getmtime(p))
        except Exception:
            # Fallback to lexicographic sort if mtime fails
            candidates.sort()

        if len(candidates) <= self.backupCount:
            return []
        # Delete oldest files beyond backupCount
        return candidates[: len(candidates) - self.backupCount]

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
log_file = os.path.join(LOG_PATH, "video_processor.log")
file_handler = CustomTimedRotatingFileHandler(
    log_file, when="midnight", interval=1, backupCount=60, encoding='utf8', utc=False
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

if DOTENV_LOADED_FROM:
    logger.info(f"Loaded environment variables from: {DOTENV_LOADED_FROM} (override=True)")
else:
    logger.info("Loaded environment variables from default dotenv search (override=True)")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)
logging.getLogger("moviepy").setLevel(logging.WARNING) # Keep moviepy logs quiet
logging.getLogger("ultralytics").setLevel(logging.WARNING) # Suppress YOLO logs

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
USERNAME = os.getenv("TELEGRAM_NOTIFY_USERNAME")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER")
OBJECT_DETECTION_MODEL_PATH = os.getenv("OBJECT_DETECTION_MODEL_PATH", default=os.path.join("models", "best_openvino_model"))
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

# Marker file to record restart timestamp
RESTART_MARKER_PATH = os.path.join(TEMP_DIR, "restart_marker.json")
# Processing ledger to track file statuses across restarts
PROCESSING_LEDGER_PATH = os.path.join(TEMP_DIR, "processing_ledger.json")
PROCESSING_LEDGER_LOCK = threading.RLock()

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

# Import after .env and logging are configured to ensure modules read env vars
from analyze_video import analyze_video
from detect_motion import detect_motion
from telegram_notification import button_callback, send_notifications, reaction_callback, cleanup_temp_media
from path_utils import parse_datetime_from_path
import re

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


# --- FileHandler (uses executor) ---
class FileHandler(FileSystemEventHandler):
    def __init__(self, loop, app):
        self.loop = loop
        self.app = app
        self.logger = logging.getLogger(__name__)

    def on_created(self, event):
        """
        Handles the 'file created' event from watchdog for .mp4 files.
        Schedules the `handle_event` coroutine to run on the event loop.

        Args:
            event (watchdog.events.FileSystemEvent): The event object from watchdog.
        """
        if event.is_directory: return
        if not event.src_path.endswith('.mp4'): return

        # Record the file immediately so restart recovery can pick it up even if we
        # restart while waiting for the file to become stable (mid-write).
        try:
            update_processing_ledger(event.src_path, "queued", {"detected_ts": time.time()})
        except Exception as e:
            # Ledger write must not break the watchdog thread.
            self.logger.warning(f"Failed to write queued ledger entry for {event.src_path}: {e}")

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
        # Build timestamp text like "HHHMM" for buttons, robust to legacy/new formats
        dt = None
        try:
            dt = parse_datetime_from_path(file_path)
        except Exception:
            dt = None
        if dt:
            timestamp_text = f"{dt.strftime('%H')}H{dt.strftime('%M')}M"
        else:
            # Fallback for unexpected names: derive HH from parent folder, MM from basename
            parent = file_path.split(os.path.sep)[-2] if os.path.sep in file_path else ""
            hh = parent[-2:] if len(parent) >= 2 and parent[-2:].isdigit() else None
            mm = None
            m_mmss = re.match(r"^(\d{2})M(\d{2})S", file_basename)
            if m_mmss:
                mm = m_mmss.group(1)
            else:
                m_digits = re.match(r"^(\d{2})(\d{2})", file_basename)
                if m_digits:
                    mm = m_digits.group(1)
            safe_hh = hh if (isinstance(hh, str) and len(hh) == 2 and hh.isdigit()) else "00"
            safe_mm = mm if (isinstance(mm, str) and len(mm) == 2 and mm.isdigit()) else "00"
            timestamp_text = f"{safe_hh}H{safe_mm}M"
        self.logger.info(f"[{file_basename}] New file detected: {file_path.split(os.path.sep)[-2]}/{file_basename}")

        # Ensure the file is represented in the ledger even if this handler is invoked
        # from restart recovery (which bypasses the watchdog on_created callback).
        update_processing_ledger(file_path, "queued", {"detected_ts": time.time()})

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
            update_processing_ledger(file_path, "started", {"start_ts": time.time()})
            
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
            update_processing_ledger(file_path, "completed", {"end_ts": time.time()})
        except Exception as e:
            self.logger.error(f"[{file_basename}] Error during video processing pipeline: {e}", exc_info=True)
            video_response = f"_{timestamp_text}:_ \u274C Відео не вдалося проаналізувати: " + str(e)[:512] + "..."
            insignificant_frames = []
            clip_path = None
            update_processing_ledger(file_path, "failed", {"end_ts": time.time(), "error": str(e)[:256]})

        battery = psutil.sensors_battery()
        if not battery.power_plugged:
            battery_time_left = time.strftime("%H:%M", time.gmtime(battery.secsleft))
            if battery.percent <= 50:
                video_response += f"\n\U0001FAAB *{battery.percent}% ~{battery_time_left}*"
                if battery.percent <= 10:
                    video_response += "\n_I don't feel so good, Mr.Stark. I don't want to go..._ \U0001F622"
            else:
                video_response += f"\n\U0001F50B *{battery.percent}% ~{battery_time_left}*"

        try:
            await send_notifications(self.app, video_response, insignificant_frames, clip_path, file_path, file_basename, timestamp_text, preserve_media_on_failure=True, allow_plain_fallback=False)
            update_processing_ledger(file_path, "completed", {"telegram_status": "sent"})
        except Exception as e_send:
            logger.error(f"[{file_basename}] Telegram send failed, scheduling retries: {e_send}")
            update_processing_ledger(file_path, "completed", {"telegram_status": "failed", "last_error": str(e_send)[:256]})
            schedule_notification_retries(
                self.app,
                video_response,
                insignificant_frames,
                clip_path,
                file_path,
                file_basename,
                timestamp_text,
                delays=(300, 600, 900)
            )
            # Do not raise; allow pipeline to finish without blocking future videos


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
        self.logger.debug("[%s] Checking file stability for: %s", file_basename, file_path)
        last_size = -1
        stable_checks = 0
        while stable_checks < checks:
            try:
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0: # Ensure size is not zero
                    stable_checks += 1
                    self.logger.debug("[%s] File size stable (%d bytes), check %d/%d.", file_basename, current_size, stable_checks, checks)
                else:
                    stable_checks = 0 # Reset if size changes or is zero
                    self.logger.debug("[%s] File size changed/zero (%d -> %d bytes). Resetting stability check.", file_basename, last_size, current_size)
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


# --- Restart Recovery Utilities ---
def write_restart_marker(path: str):
    try:
        payload = {"restart_requested_at": time.time()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        logger.info(f"Wrote restart marker: {path}")
    except Exception as e:
        logger.error(f"Failed to write restart marker '{path}': {e}")


def read_restart_marker(path: str):
    try:
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ts = float(data.get("restart_requested_at", 0))
        return ts if ts > 0 else None
    except Exception as e:
        logger.error(f"Failed to read restart marker '{path}': {e}")
        return None


def read_processing_ledger(path: str):
    with PROCESSING_LEDGER_LOCK:
        try:
            if not os.path.exists(path):
                return {}
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}


def write_processing_ledger(path: str, ledger: dict):
    with PROCESSING_LEDGER_LOCK:
        try:
            tmp_path = path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(ledger, f)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"Failed to write processing ledger '{path}': {e}")


def prune_processing_ledger(ledger: dict, max_entries: int = 100) -> dict:
    """Keep only the last `max_entries` entries by most recent timestamp (end_ts/start_ts)."""
    try:
        items = []
        for k, v in ledger.items():
            ts = v.get("end_ts") or v.get("start_ts") or 0
            items.append((ts, k, v))
        items.sort(key=lambda x: x[0], reverse=True)
        pruned = items[:max_entries]
        return {k: v for _, k, v in pruned}
    except Exception as e:
        logger.error(f"Failed to prune processing ledger: {e}")
        return ledger


# --- Retention Cleanup ---
def cleanup_old_videos(root: str, days: int, logger: logging.Logger) -> int:
    """Delete .mp4 files older than `days` under `root`. Returns count deleted."""
    cutoff = time.time() - days * 86400
    deleted = 0
    try:
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.lower().endswith('.mp4'):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    mtime = os.path.getmtime(fp)
                    if mtime < cutoff:
                        os.remove(fp)
                        deleted += 1
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"Retention cleanup failed: {e}")
    logger.info(f"Retention cleanup done. Deleted {deleted} old video(s) (> {days}d).")
    return deleted


def cleanup_empty_folders(root: str, logger: logging.Logger) -> int:
    """Remove empty directories under `root` (bottom-up). Returns count removed."""
    removed = 0
    try:
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            # Do not remove the root folder itself
            if os.path.abspath(dirpath) == os.path.abspath(root):
                continue
            try:
                # If there are any files, not empty
                if filenames:
                    continue
                # If any subdir still exists, not empty
                if dirnames:
                    continue
                os.rmdir(dirpath)
                removed += 1
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"Empty folder cleanup failed: {e}")
    if removed:
        logger.info(f"Empty folder cleanup done. Removed {removed} empty folder(s).")
    return removed


def cleanup_old_temp_days(temp_root: str, days: int, logger: logging.Logger) -> int:
    """Remove temp_root/YYYYMMDD directories older than `days` (recursive). Returns count removed."""
    removed = 0
    try:
        if not os.path.isdir(temp_root):
            return 0
        cutoff_date = datetime.now().date() - timedelta(days=int(days))
        for name in os.listdir(temp_root):
            if not re.fullmatch(r"\d{8}", name):
                continue
            day_path = os.path.join(temp_root, name)
            if not os.path.isdir(day_path):
                continue
            try:
                day = datetime.strptime(name, "%Y%m%d").date()
            except Exception:
                continue
            if day < cutoff_date:
                try:
                    shutil.rmtree(day_path, ignore_errors=False)
                    removed += 1
                except Exception:
                    # If partial removal fails, keep going
                    continue
    except Exception as e:
        logger.warning(f"Temp daily cleanup failed: {e}")
    if removed:
        logger.info(f"Temp daily cleanup done. Removed {removed} day folder(s) (> {days}d).")
    return removed


def run_retention_cleanup(video_root: str, video_days: int, temp_root: str, temp_days: int, logger: logging.Logger) -> None:
    """Run all retention-related cleanup tasks."""
    try:
        cleanup_old_videos(video_root, video_days, logger)
        cleanup_empty_folders(video_root, logger)
        cleanup_old_temp_days(temp_root, temp_days, logger)
    except Exception as e:
        logger.warning(f"Retention cleanup job failed: {e}")


async def retention_scheduler(stop_event: asyncio.Event, root: str, days: int, logger: logging.Logger):
    """Run daily at configured time to remove old videos."""
    # Parse HH:MM
    try:
        hh, mm = [int(x) for x in RETENTION_CLEANUP_TIME.split(':')]
    except Exception:
        hh, mm = 0, 10
    while not stop_event.is_set():
        now = datetime.now()
        target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        wait_secs = (target - now).total_seconds()
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=wait_secs)
            if stop_event.is_set():
                break
        except asyncio.TimeoutError:
            pass
        # Run cleanup (in thread executor to avoid blocking)
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, run_retention_cleanup, VIDEO_FOLDER, days, TEMP_DIR, TEMP_RETENTION_DAYS, logger)
        except Exception as e:
            logger.warning(f"Scheduled retention cleanup failed: {e}")


def update_processing_ledger(file_path: str, status: str, extra: dict | None = None):
    with PROCESSING_LEDGER_LOCK:
        ledger = read_processing_ledger(PROCESSING_LEDGER_PATH)
        entry = ledger.get(file_path, {})

        prev_status = entry.get("status")
        # Never downgrade a completed entry; allow completed->completed updates.
        if prev_status == "completed" and status != "completed":
            if extra:
                entry.update(extra)
                ledger[file_path] = entry
                ledger = prune_processing_ledger(ledger, 100)
                write_processing_ledger(PROCESSING_LEDGER_PATH, ledger)
            return

        # Avoid downgrading started->queued (can happen on duplicate FS events).
        if prev_status == "started" and status == "queued":
            status = "started"

        entry.update({"status": status})
        if extra:
            entry.update(extra)
        ledger[file_path] = entry
        ledger = prune_processing_ledger(ledger, 100)
        write_processing_ledger(PROCESSING_LEDGER_PATH, ledger)


def schedule_notification_retries(app,
                                  video_response: str,
                                  insignificant_frames: list,
                                  clip_path: str | None,
                                  file_path: str,
                                  file_basename: str,
                                  timestamp_text: str,
                                  delays: tuple[int, ...] = (300, 600, 900)):
    async def retry_sender():
        for delay in delays:
            try:
                await asyncio.sleep(delay)
                await send_notifications(app, video_response, insignificant_frames, clip_path, file_path, file_basename, timestamp_text, preserve_media_on_failure=True, allow_plain_fallback=False)
                logger.info(f"[{file_basename}] Telegram retry succeeded after {delay} seconds.")
                update_processing_ledger(file_path, "completed", {"telegram_status": "sent_retry", "retry_delay": delay, "end_ts": time.time()})
                return
            except Exception as e:
                logger.error(f"[{file_basename}] Telegram retry failed after {delay} seconds: {e}")
                update_processing_ledger(file_path, "completed", {"telegram_status": "retry_failed", "last_error": str(e)[:256]})
                continue
        logger.warning(f"[{file_basename}] Telegram retries exhausted; attempting plain message.")
        try:
            # Final attempt: allow plain fallback
            await send_notifications(app, video_response, insignificant_frames, clip_path, file_path, file_basename, timestamp_text, preserve_media_on_failure=True, allow_plain_fallback=True)
            update_processing_ledger(file_path, "completed", {"telegram_status": "sent_plain_after_retries", "end_ts": time.time()})
        except Exception as e_final:
            logger.error(f"[{file_basename}] Final plain message attempt failed: {e_final}")
            update_processing_ledger(file_path, "failed", {"telegram_status": "final_plain_failed", "last_error": str(e_final)[:256]})
        try:
            if clip_path:
                await cleanup_temp_media(clip_path, file_path, logger, file_basename)
        except Exception as e_clean:
            logger.warning(f"[{file_basename}] Cleanup after exhausted retries failed: {e_clean}")

    try:
        asyncio.create_task(retry_sender(), name=f"TelegramRetry-{file_basename}")
    except Exception:
        # Fallback: run without named task
        asyncio.create_task(retry_sender())


async def recover_missed_files_since(since_ts: float):
    """
    Scan the recent hour directories for .mp4 files modified between
    (since_ts - RESTART_RECOVERY_WINDOW_SECONDS) and now, and process them.

    Runs after the Telegram application is started to ensure notifications work.
    """
    try:
        # Wait until Telegram application is running
        for _ in range(50):  # up to ~10s
            if getattr(application, "running", False):
                break
            await asyncio.sleep(0.2)

        window_start = max(0.0, since_ts - RESTART_RECOVERY_WINDOW_SECONDS)
        window_end = time.time()

        start_dt = datetime.fromtimestamp(window_start)
        end_dt = datetime.fromtimestamp(window_end)

        # Generate candidate hour dir names between start_dt and end_dt
        candidate_dirs = set()
        cursor = start_dt.replace(minute=0, second=0, microsecond=0)
        while cursor <= end_dt:
            candidate_dirs.add(cursor.strftime("%Y%m%d%H"))
            cursor += timedelta(hours=1)

        ledger = read_processing_ledger(PROCESSING_LEDGER_PATH)
        files_to_process = []
        seen_paths = set()
        mtime_candidates = 0
        day_mtime_candidates = 0
        fallback_mtime_candidates = 0
        ledger_candidates = 0
        day_dirs_found = 0
        for hour_dir in candidate_dirs:
            dir_path = os.path.join(VIDEO_FOLDER, hour_dir)
            if not os.path.isdir(dir_path):
                continue
            try:
                for name in os.listdir(dir_path):
                    if not name.endswith(".mp4"):
                        continue
                    fpath = os.path.join(dir_path, name)
                    try:
                        mtime = os.path.getmtime(fpath)
                    except Exception:
                        continue
                    if window_start <= mtime <= window_end:
                        entry = ledger.get(fpath)
                        if entry and entry.get("status") == "completed":
                            continue
                        if fpath not in seen_paths:
                            files_to_process.append((mtime, fpath))
                            seen_paths.add(fpath)
                            mtime_candidates += 1
            except Exception as e:
                logger.error(f"Error scanning directory '{dir_path}': {e}")

        # Primary layout scan: VIDEO_FOLDER/YYYY/MM/DD/<video>.mp4
        # Scan only the day folders that overlap the recovery window.
        try:
            day_cursor = start_dt.date()
            while day_cursor <= end_dt.date():
                day_path = os.path.join(
                    VIDEO_FOLDER,
                    f"{day_cursor.year:04d}",
                    f"{day_cursor.month:02d}",
                    f"{day_cursor.day:02d}",
                )
                if os.path.isdir(day_path):
                    day_dirs_found += 1
                    try:
                        for name in os.listdir(day_path):
                            if not name.endswith(".mp4"):
                                continue
                            fpath = os.path.join(day_path, name)
                            try:
                                mtime = os.path.getmtime(fpath)
                            except Exception:
                                continue
                            if not (window_start <= mtime <= window_end):
                                continue
                            entry = ledger.get(fpath)
                            if entry and entry.get("status") == "completed":
                                continue
                            if fpath in seen_paths:
                                continue
                            files_to_process.append((mtime, fpath))
                            seen_paths.add(fpath)
                            day_mtime_candidates += 1
                    except Exception as e:
                        logger.error(f"Error scanning directory '{day_path}': {e}")
                day_cursor += timedelta(days=1)
        except Exception as e:
            logger.error(f"Restart recovery day-scan failed: {e}")

        # Fallback: if the storage layout isn't VIDEO_FOLDER/YYYYMMDDHH/, the hour-dir scan
        # can miss everything (e.g., VIDEO_FOLDER/<DD>/file.mp4). Do a bounded recursive scan
        # under VIDEO_FOLDER to catch recent files by mtime.
        # IMPORTANT: do NOT do this just because the window is quiet; only do it if we
        # couldn't find any expected day directories at all (layout mismatch).
        if mtime_candidates == 0 and day_mtime_candidates == 0 and day_dirs_found == 0:
            try:
                max_depth = 3
                root_parts = os.path.abspath(VIDEO_FOLDER).rstrip(os.path.sep).split(os.path.sep)
                for dirpath, dirnames, filenames in os.walk(VIDEO_FOLDER, topdown=True):
                    abs_dirpath = os.path.abspath(dirpath)
                    parts = abs_dirpath.rstrip(os.path.sep).split(os.path.sep)
                    depth = max(0, len(parts) - len(root_parts))
                    if depth >= max_depth:
                        dirnames[:] = []

                    for name in filenames:
                        if not name.endswith(".mp4"):
                            continue
                        fpath = os.path.join(dirpath, name)
                        try:
                            mtime = os.path.getmtime(fpath)
                        except Exception:
                            continue
                        if not (window_start <= mtime <= window_end):
                            continue
                        entry = ledger.get(fpath)
                        if entry and entry.get("status") == "completed":
                            continue
                        if fpath in seen_paths:
                            continue
                        files_to_process.append((mtime, fpath))
                        seen_paths.add(fpath)
                        fallback_mtime_candidates += 1
            except Exception as e:
                logger.error(f"Restart recovery fallback scan failed: {e}")

        # Include ALL files that are started/failed in the ledger (regardless of start_ts), as long as they exist.
        # This ensures we don't miss in-progress items that began before the recovery window.
        try:
            for fpath, entry in ledger.items():
                if entry.get("status") == "completed":
                    continue
                if not os.path.exists(fpath):
                    continue
                if fpath in seen_paths:
                    continue
                # Use ledger start_ts if available; otherwise fall back to file mtime, else window_start
                ts = float(entry.get("start_ts") or 0) or os.path.getmtime(fpath) or window_start
                files_to_process.append((ts, fpath))
                seen_paths.add(fpath)
                ledger_candidates += 1
        except Exception as e:
            logger.error(f"Error merging ledger entries for recovery: {e}")

        if not files_to_process:
            logger.info("Restart recovery: No missed files found in window.")
            return

        files_to_process.sort(key=lambda x: x[0])
        logger.info(f"Restart recovery: Found {len(files_to_process)} file(s) to process.")

        # Use FileHandler directly to run the standard pipeline
        loop = asyncio.get_running_loop()
        fh = FileHandler(loop, application)
        processed_count = 0
        failed_count = 0
        for _, fpath in files_to_process:
            try:
                event = SimpleNamespace(src_path=fpath, is_directory=False)
                await fh.handle_event(event)
                logger.info(f"Restart recovery: Processed {fpath}")
                processed_count += 1
            except Exception as e:
                logger.error(f"Restart recovery: Failed processing {fpath}: {e}", exc_info=True)
                failed_count += 1

        logger.info(
            f"Restart recovery summary: candidates={len(files_to_process)} (mtime={mtime_candidates}, day_mtime={day_mtime_candidates}, fallback_mtime={fallback_mtime_candidates}, ledger={ledger_candidates}), "
            f"processed={processed_count}, failed={failed_count}"
        )
    except Exception as e:
        logger.error(f"Restart recovery task failed: {e}", exc_info=True)


# Add the callback handler
application.add_handler(CallbackQueryHandler(button_callback))
application.add_handler(MessageReactionHandler(reaction_callback))


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
        await application.updater.start_polling(poll_interval=1.0, timeout=20, allowed_updates=["message_reaction", "callback_query"])
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

async def run_main_script_watcher(stop_event, watch_dir):
    """
    Watches SCRIPT_DIR for any top-level Python file (*.py) changes (non-recursive).
    On change, sets `stop_event` to trigger a graceful restart.

    Args:
        stop_event (asyncio.Event): Global stop event to set on change.
        watch_dir (str): Absolute path to the directory to watch (non-recursive).
    """
    watch_dir = os.path.abspath(watch_dir)
    logger.info(f"Starting self-watcher for directory: {watch_dir} (*.py, non-recursive)")
    observer = None
    try:
        event_handler = MainScriptChangeHandler(stop_event, watch_dir)
        observer = Observer()
        observer.schedule(event_handler, path=watch_dir, recursive=False)
        observer.start()
        logger.info("Self-watcher started.")

        while not stop_event.is_set():
            if not observer.is_alive():
                logger.error("Self-watcher observer thread died unexpectedly.")
                stop_event.set()
                break
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        logger.info("Main script watcher task cancelled.")
    except Exception as e:
        logger.error(f"Error in Main script watcher task: {e}", exc_info=True)
        stop_event.set()
    finally:
        logger.info("Stopping main script watcher...")
        if observer and observer.is_alive():
            observer.stop()
            try:
                observer.join(timeout=2.0)
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
    # Optionally start the log dashboard in a background thread
    dashboard_runner = None
    if ENABLE_LOG_DASHBOARD:
        try:
            from tools.log_dashboard.app import app as log_dashboard_app
            dashboard_runner = DashboardRunner(log_dashboard_app, LOG_DASHBOARD_HOST, LOG_DASHBOARD_PORT, log_level="info")
            dashboard_runner.start()
        except Exception as e:
            logger.error(f"Failed to start log dashboard: {e}")
    global RESTART_REQUESTED # Allow main to modify it if needed, though not strictly necessary here
    RESTART_REQUESTED = False # Ensure it's reset if main is somehow called again in same process (unlikely)

    # Schedule daily retention cleanup
    asyncio.create_task(retention_scheduler(stop_event, VIDEO_FOLDER, RETENTION_DAYS, logger))
    # Use task names for better debugging if needed
    telegram_task = asyncio.create_task(run_telegram_bot(stop_event), name="TelegramBotTask")
    watcher_task = asyncio.create_task(run_file_watcher(stop_event), name="FileWatcherTask")
    main_script_monitor_task = asyncio.create_task(
        run_main_script_watcher(stop_event, SCRIPT_DIR),
        name="MainScriptWatcherTask"
    )

    tasks = {telegram_task, watcher_task, main_script_monitor_task}
    recovery_task_ref = None

    # If a restart marker exists, schedule recovery of missed files
    marker_ts = read_restart_marker(RESTART_MARKER_PATH)
    if marker_ts is not None:
        # Only recover for recent restarts within a reasonable window (e.g., 10 minutes)
        if (time.time() - marker_ts) <= max(600, RESTART_RECOVERY_WINDOW_SECONDS * 2):
            logger.info("Restart marker detected. Scheduling missed-file recovery...")
            recovery_task_ref = asyncio.create_task(recover_missed_files_since(marker_ts), name="MissedFilesRecoveryTask")
            try:
                recovery_task_ref.add_done_callback(lambda t: logger.info("MissedFilesRecoveryTask completed."))
            except Exception:
                pass
        else:
            logger.info("Restart marker is too old. Skipping recovery.")
        try:
            os.remove(RESTART_MARKER_PATH)
        except Exception:
            pass

    # Add log dashboard watcher to auto-restart the dashboard on changes
    if ENABLE_LOG_DASHBOARD and dashboard_runner is not None:
        try:
            log_dashboard_dir = os.path.join(SCRIPT_DIR, "tools", "log_dashboard")
            if os.path.isdir(log_dashboard_dir):
                observer = Observer()
                observer.schedule(LogDashboardChangeHandler(dashboard_runner), path=log_dashboard_dir, recursive=True)
                observer.start()
                logger.info(f"Watching log dashboard directory for changes: {log_dashboard_dir}")

                async def _monitor_dashboard_observer(stop_event):
                    try:
                        while not stop_event.is_set():
                            if not observer.is_alive():
                                logger.error("Log dashboard watcher died unexpectedly.")
                                break
                            await asyncio.sleep(1)
                    finally:
                        try:
                            observer.stop()
                            observer.join(timeout=2.0)
                        except Exception:
                            pass

                dashboard_watch_task = asyncio.create_task(_monitor_dashboard_observer(stop_event), name="LogDashboardWatcher")
                tasks.add(dashboard_watch_task)
        except Exception as e:
            logger.error(f"Failed to initialize log dashboard watcher: {e}")
    logger.info("Application started. Press Ctrl+C to exit. Will auto-restart on any *.py change in SCRIPT_DIR.")

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

        # Ensure recovery task is not left hanging
        if recovery_task_ref is not None and not recovery_task_ref.done():
            try:
                recovery_task_ref.cancel()
                await asyncio.gather(recovery_task_ref, return_exceptions=True)
            except Exception:
                pass

        # Stop dashboard if running
        if dashboard_runner is not None:
            try:
                dashboard_runner.stop()
                logger.info("Log dashboard stopped.")
            except Exception as e_dash:
                logger.error(f"Error stopping log dashboard: {e_dash}")

        if RESTART_REQUESTED:
            logger.info("Fast shutdown for restart: Not waiting for current analysis to finish.")
            # Shutdown immediately without waiting for the worker.
            motion_executor.shutdown(wait=False, cancel_futures=True)
            io_executor.shutdown(wait=False, cancel_futures=True)
            logger.info("Executors issued fast shutdown command.")

            logger.info("RESTART_REQUESTED is True. Executing self-restart...")
            # Write restart marker before replacing the process
            write_restart_marker(RESTART_MARKER_PATH)
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
