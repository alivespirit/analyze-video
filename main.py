import os
import datetime
import asyncio
import concurrent.futures
import logging
import re
import sys
import time

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
from moviepy import VideoFileClip, concatenate_videoclips

import threading

GEMINI_CONCURRENCY_LIMIT = 1  # Only one Gemini call at a time (adjust if safe)
gemini_semaphore = threading.Semaphore(GEMINI_CONCURRENCY_LIMIT)

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
# Use min(32, os.cpu_count() + 4) as suggested by ThreadPoolExecutor docs
# or stick to a reasonable number if cpu_count is unreliable/very large
max_workers = min(32, (os.cpu_count() or 1) + 4)
try:
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    logger.info(f"ThreadPoolExecutor initialized with max_workers={max_workers}.")
except Exception as e:
     logger.critical(f"Failed to initialize ThreadPoolExecutor: {e}", exc_info=True)
     exit(1)


# --- analyze_video function (remains synchronous, uses original models) ---
def analyze_video(video_path):
    """Extract insights from the video using Gemini. Runs in executor."""
    file_basename = os.path.basename(video_path)
    timestamp = f"_{file_basename[:6]}:_ "

    video_bytes = open(video_path, 'rb').read()

    model_main = 'gemini-2.5-flash'
    model_fallback = 'gemini-2.5-flash-lite-preview-06-17'
    model_fallback_2_0 = 'gemini-2.0-flash'
    model_pro = 'gemini-2.5-pro-exp-03-25'

    sampling_rate = 5  # sampling rate in FPS
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

        analysis_result = ""
        additional_text = ""

        for attempt in range(max_retries):
            try:
                with gemini_semaphore:
                    logger.info(f"[{file_basename}] Generating content ({model_main}), attempt {attempt+1}...")
                    response = client.models.generate_content(
                                  model=model_main,
                                  contents=types.Content(
                                      parts=[
                                          types.Part(
                                              inline_data=types.Blob(
                                                  data=video_bytes,
                                                  mime_type='video/mp4'),
                                              video_metadata=types.VideoMetadata(fps=sampling_rate)
                                          ),
                                          types.Part(text=prompt)
                                      ]
                                  ),
                                  config=types.GenerateContentConfig(
                                      automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                          disable=True
                                      )
                                  )
                              )
                logger.info(f"[{file_basename}] {model_main} response received.")
                analysis_result = response.text
                break
            except Exception as e_flash:
                try:
                    with gemini_semaphore:
                        logger.warning(f"[{file_basename}] {model_main} failed. Falling back to {model_fallback}. Message: {e_flash}")
                        response = client.models.generate_content(
                                      model=model_fallback,
                                      contents=types.Content(
                                          parts=[
                                              types.Part(
                                                  inline_data=types.Blob(
                                                      data=video_bytes,
                                                      mime_type='video/mp4'),
                                                  video_metadata=types.VideoMetadata(fps=sampling_rate)
                                              ),
                                              types.Part(text=prompt)
                                          ]
                                      ),
                                      config=types.GenerateContentConfig(
                                          automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                              disable=True
                                          )
                                      )
                                  )
                    logger.info(f"[{file_basename}] {model_fallback} response received.")
                    analysis_result = "_[2.5FL]_ " + response.text
                    break
                except Exception as e_flash_2_5_fl:
                    logger.warning(f"[{file_basename}] {model_fallback} also failed: {e_flash_2_5_fl}")
                    if attempt < max_retries - 1:
                        wait_time = 10 * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s
                        logger.warning(f"[{file_basename}] Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        try:
                            logger.info(f"[{file_basename}] Attempting {model_fallback_2_0} as final fallback...")
                            response = client.models.generate_content(
                                          model=model_fallback_2_0,
                                          contents=types.Content(
                                              parts=[
                                                  types.Part(
                                                      inline_data=types.Blob(
                                                          data=video_bytes,
                                                          mime_type='video/mp4'),
                                                      video_metadata=types.VideoMetadata(fps=sampling_rate)
                                                  ),
                                                  types.Part(text=prompt)
                                              ]
                                          ),
                                          config=types.GenerateContentConfig(
                                              automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                                  disable=True
                                              )
                                          )
                                      )
                            logger.info(f"[{file_basename}] {model_fallback_2_0} response received.")
                            analysis_result = "_[2.0]_ " + response.text
                            break
                        except Exception as e_flash_2_0:
                            logger.error(f"[{file_basename}] {model_fallback_2_0} failed as well: {e_flash_2_0}")
                            logger.error(f"[{file_basename}] Giving up after retries.")
                            raise # Re-raise the exception to handle it in the outer scope

        logger.info(f"[{file_basename}] Response: {analysis_result}")

        now = datetime.datetime.now()
        # Disabled 2.5 Pro for now, as it has no free tier quota
        if False: #("Отакої!" in analysis_result) and (9 <= now.hour <= 13):
            logger.info(f"[{file_basename}] Trying {model_pro}...")
            try:
                response_new = client.models.generate_content(
                                  model=model_pro,
                                  contents=types.Content(
                                      parts=[
                                          types.Part(
                                              inline_data=types.Blob(
                                                  data=video_bytes,
                                                  mime_type='video/mp4'),
                                              video_metadata=types.VideoMetadata(fps=sampling_rate)
                                          ),
                                          types.Part(text=prompt)
                                      ]
                                  ),
                                  config=types.GenerateContentConfig(
                                      automatic_function_calling=types.AutomaticFunctionCallingConfig(
                                          disable=True
                                      )
                                  )
                              )
                logger.info(f"[{file_basename}] {model_pro} response received.")
                logger.info(f"[{file_basename}] {response_new.text}")
                additional_text = "\n_[2.5 Pro]_ " + response_new.text + "\n" + USERNAME
            except Exception as e_pro:
                # Log as warning since Flash result is still available
                logger.warning(f"[{file_basename}] Error in {model_pro}: {e_pro}", exc_info=True)
                additional_text = "\n_[2.5 Pro]_ Failed.\n" + USERNAME
        if ("Отакої!" in analysis_result) and (9 <= now.hour <= 13):
            additional_text = "\n" + USERNAME

        return timestamp + analysis_result + additional_text

    except Exception as e_analysis:
        logger.error(f"[{file_basename}] Video analysis failed: {e_analysis}", exc_info=True)
        return timestamp + "Video analysis failed."


def time_to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

def extract_video_clip(video_path, timeranges):
    """
    Extracts multiple subclips from a video and concatenates them into one video.
    :param video_path: Path to the source video.
    :param timeranges: List of strings in format ["MM:SS-MM:SS", ...].
    :return: Path to the output video file in the local temp/ directory.
    """
    try:
        with VideoFileClip(video_path) as video:
            subclips = []
            for tr in timeranges:
                start_str, end_str = tr.split('-')
                start_sec = time_to_seconds(start_str)
                end_sec = time_to_seconds(end_str)
                subclip = video.subclipped(start_sec, end_sec)
                subclips.append(subclip)
            if not subclips:
                raise ValueError("No valid timeranges provided.")
            final_clip = concatenate_videoclips(subclips)
            # Use the local TEMP_DIR instead of the system's temp directory
            timeranges_str = "_".join([tr.replace(":", "").replace("-", "to").replace("00", "") for tr in timeranges])
            output_mp4_path = os.path.join(
                TEMP_DIR,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_{timeranges_str}.mp4"
            )
            final_clip.write_videofile(
                output_mp4_path,
                codec='libx264',
                audio=False,
                bitrate='1500k',
                preset='medium',
                threads=4,
                logger=None
            )
        return output_mp4_path
    except Exception as e:
        logger.error(f"Error extracting and concatenating video clips: {e}", exc_info=True)
        return None

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
        file_path = event.src_path
        file_basename = os.path.basename(file_path)
        self.logger.info(f"[{file_basename}] New file detected: {file_path}")

        try:
            await self.wait_for_file_stable(file_path, file_basename)
        except FileNotFoundError:
            self.logger.warning(f"[{file_basename}] File disappeared before analysis could start: {file_path}")
            return
        except Exception as e_wait:
            self.logger.error(f"[{file_basename}] Error waiting for file stability: {e_wait}", exc_info=True)
            return

        try:
            current_loop = asyncio.get_running_loop()
            video_response = await current_loop.run_in_executor(
                executor, analyze_video, file_path
            )
            self.logger.info(f"[{file_basename}] Analysis complete.")
        except Exception as e:
            self.logger.error(f"[{file_basename}] Error running analyze_video in executor: {e}", exc_info=True)
            video_response = f"_{file_basename[:6]}:_ Failed to analyze video."

        # --- MODIFIED: Telegram Sending Logic ---
        try:
            safe_video_folder = os.path.join(VIDEO_FOLDER, '')
            if file_path.startswith(safe_video_folder):
                 callback_file = file_path[len(safe_video_folder):].replace(os.path.sep, '/')
            else:
                 self.logger.warning(f"[{file_basename}] File path '{file_path}' does not start with VIDEO_FOLDER '{safe_video_folder}'. Using basename for callback.")
                 callback_file = file_basename

            keyboard = [[InlineKeyboardButton("Глянути", callback_data=callback_file)]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            def format_caption(text):
                # Remove timestamps from response text
                return re.sub(r"(\s+)(\d{2}:\d{2}-\d{2}:\d{2}),?(\s+)", " ", text)

            send_plain_message = True
            if "Отакої!" in video_response:
                # Regex to find MM:SS-MM:SS
                time_marker_regex = r"\s+(\d{2}:\d{2}-\d{2}:\d{2}),?\s+"
                matches = re.findall(time_marker_regex, video_response)

                if matches:
                    index_text = f"\n`{matches}`"
                    media_path = None
                    try:
                        self.logger.info(f"[{file_basename}] Extracting video clip for ranges {matches}...")
                        media_path = extract_video_clip(file_path, matches) 
                        if media_path:
                            self.logger.info(f"[{file_basename}] Video clip extracted: {media_path}")
                            self.logger.info(f"[{file_basename}] Sending animation (MP4) with button to Telegram...")
                            with open(media_path, 'rb') as animation_file:
                                await self.app.bot.send_animation(
                                    chat_id=CHAT_ID,
                                    animation=animation_file,
                                    caption=format_caption(video_response) + index_text,
                                    reply_markup=reply_markup,
                                    parse_mode='Markdown'
                                )
                            self.logger.info(f"[{file_basename}] Animation sent successfully.")
                            send_plain_message = False

                    except telegram.error.BadRequest as bad_request_error:
                        self.logger.warning(f"[{file_basename}] BadRequest error: {bad_request_error}. Retrying with escaped Markdown.")
                        try:
                            with open(media_path, 'rb') as animation_file:
                                await self.app.bot.send_animation(chat_id=CHAT_ID, animation=animation_file, caption=escape_markdown(video_response, version=1), reply_markup=reply_markup, parse_mode='Markdown')
                            self.logger.info(f"[{file_basename}] Media sent successfully after escaping Markdown.")
                            send_plain_message = False
                        except Exception as retry_error:
                            self.logger.error(f"[{file_basename}] Failed to send media after escaping Markdown: {retry_error}", exc_info=True)
                    except Exception as e:
                        self.logger.error(f"[{file_basename}] Error sending media: {e}", exc_info=True)
                    finally:
                        # Delete the generated clip after sending or if an error occurs
                        if media_path and os.path.exists(media_path):
                            await asyncio.sleep(30) # Give a moment before deleting
                            os.remove(media_path)
                            self.logger.info(f"[{file_basename}] Temporary media file deleted: {media_path}")

                    if send_plain_message: # This will be true if media creation or sending failed
                        self.logger.warning(f"[{file_basename}] Failed to create/send media for {matches}. Sending message instead.")
                else:
                    self.logger.warning(f"[{file_basename}] No valid time marker found in response. Sending message instead.")

            if send_plain_message:
                self.logger.info(f"[{file_basename}] Sending message with button to Telegram...")
                try:
                    await self.app.bot.send_message(
                        chat_id=CHAT_ID,
                        text=format_caption(video_response),
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                    self.logger.info(f"[{file_basename}] Message with button sent successfully.")
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

        except Exception as e:
            self.logger.error(f"[{file_basename}] Failed to send message with button to Telegram: {e}", exc_info=True)
        finally:
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
            await query.edit_message_text(text=f"{query.message.text}\n\n_Відео файл не знайдено._", parse_mode='Markdown')
        except Exception as edit_e:
            logger.error(f"[{file_basename}] Error editing message for not found file: {edit_e}", exc_info=True)
        return

    logger.info(f"[{file_basename}] Sending video from callback...")
    try:
        with open(file_path, 'rb') as video_file:
            await context.bot.send_video(
                chat_id=query.message.chat_id, video=video_file, parse_mode='Markdown',
                caption="Осьо відео", reply_to_message_id=query.message.message_id
            )
        logger.info(f"[{file_basename}] Video sent successfully from callback.")
    except FileNotFoundError:
         logger.error(f"[{file_basename}] Video file disappeared before sending from callback: {file_path}")
         try: await query.edit_message_text(text=f"{query.message.text}\n\n_Помилка: Відео файл зник._", parse_mode='Markdown')
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

        logger.info("Shutting down thread pool executor (allowing current analysis to finish)...")
        executor.shutdown(wait=True, cancel_futures=False) # wait=True ensures running tasks finish
        logger.info("Executor shut down.")

        if RESTART_REQUESTED:
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