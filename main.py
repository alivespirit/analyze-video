import os
import time
import datetime
import asyncio
from random import randint
import concurrent.futures # Needed for executor
import logging # <<< IMPORT LOGGING

from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CallbackQueryHandler
import google.generativeai as genai

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

load_dotenv()  ## load all the environment variables

LOG_PATH = os.getenv("LOG_PATH", default="")

# --- Logging Setup ---
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
log_file = LOG_PATH + 'video_processor.log'

# File Handler
file_handler = logging.FileHandler(log_file, mode='a', encoding='utf8') # Append mode
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO) # Log INFO level and above to file

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
console_handler.setLevel(logging.INFO) # Log INFO level and above to console

# Get the root logger and add handlers
logger = logging.getLogger()
logger.setLevel(logging.INFO) # Set root logger level
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# ---------------------

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram.ext").setLevel(logging.WARNING)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
USERNAME = os.getenv("TELEGRAM_NOTIFY_USERNAME")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER")

# --- Check Environment Variables ---
if not all([GEMINI_API_KEY, TELEGRAM_TOKEN, CHAT_ID, USERNAME, VIDEO_FOLDER]):
    logger.critical("ERROR: One or more essential environment variables are missing. Exiting.")
    exit(1) # Exit if critical env vars are missing
if not os.path.isdir(VIDEO_FOLDER):
    logger.critical(f"ERROR: VIDEO_FOLDER '{VIDEO_FOLDER}' does not exist or is not a directory. Exiting.")
    exit(1)
# -----------------------------------

try:
    genai.configure(api_key=GEMINI_API_KEY)
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
    video_file = None
    try:
        logger.info(f"[{file_basename}] Uploading video to Gemini...")
        video_file = genai.upload_file(path=video_path)
    except Exception as e:
        logger.error(f"[{file_basename}] Video upload failed: {e}", exc_info=True) # Log traceback
        return timestamp + "Video upload failed."

    logger.info(f"[{file_basename}] Completed upload: {video_file.uri}")

    try:
        while video_file.state.name == "PROCESSING":
            logger.info(f"[{file_basename}] Waiting for Gemini processing...")
            time.sleep(10)
            video_file = genai.get_file(video_file.name) # Refresh state

        if video_file.state.name == "FAILED":
            logger.error(f"[{file_basename}] Video processing failed state.")
            return timestamp + "Video processing failed."

        prompt = """This video is from the surveillance camera, located on the second floor above the entrance to the building.
        The camera is facing the street. Recording is triggered by motion. Audio is unrelated and should be ignored.
        Analyze the video. Focus on the foreground with a fence and a gate, and a sidewalk. Take note of vehicles, people, and pets.
        Describe the main action in the video in 1-2 sentences. If there are multiple actions, briefly describe all of them.
        If any dog is present in the video, describe it, include keyword "*Отакої!*" and state confidence level.
        If any person is going through the gate, include keyword "*Отакої!*" and state confidence level.
        If someone is doing something to the parked red Tesla (not just passing by), describe what they are doing and include keyword "Хм...".
        Only return the output and nothing else. Respond in Ukrainian.
        """

        model_flash = genai.GenerativeModel(model_name="models/gemini-2.5-flash-preview-04-17")
        analysis_result = ""
        additional_text = ""

        logger.info(f"[{file_basename}] Generating content (2.5 Flash)...")
        try:
            response = model_flash.generate_content([prompt, video_file],
                      request_options={"timeout": 600})
            analysis_result = response.text
            logger.info(f"[{file_basename}] Gemini 2.5 Flash response received.")
        except Exception as e_flash:
            logger.warning(f"[{file_basename}] Gemini 2.5 Flash failed: {e_flash}. Falling back to Gemini 2.0 Flash.", exc_info=True)
            try:
                model_flash_2_0 = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
                response = model_flash_2_0.generate_content([prompt, video_file],
                              request_options={"timeout": 600})
                logger.info(f"[{file_basename}] Gemini 2.0 Flash response received.")
                analysis_result = "_[2.0]_ " + response.text
            except Exception as e_flash_2_0:
                logger.error(f"[{file_basename}] Gemini 2.0 Flash also failed: {e_flash_2_0}", exc_info=True)
                raise  # Re-raise the exception to handle it in the outer scope

        logger.info(f"[{file_basename}] {analysis_result}")

        now = datetime.datetime.now()
        # Consider Gemini Pro 2.5 experimental, log potential issues
        if ("Отакої!" in analysis_result) and (9 <= now.hour <= 15):
            model_pro = genai.GenerativeModel(model_name="models/gemini-2.5-pro-exp-03-25")
            logger.info(f"[{file_basename}] Trying Gemini 2.5 Pro...")
            try:
                response_new = model_pro.generate_content([prompt, video_file],
                                                          request_options={"timeout": 600})
                logger.info(f"[{file_basename}] Gemini Pro 2.5 response received.")
                logger.info(f"[{file_basename}] {response_new.text}")
                additional_text = "\n_Gemini 2.5 Pro:_ " + response_new.text + "\n" + USERNAME
            except Exception as e_pro:
                # Log as warning since Flash result is still available
                logger.warning(f"[{file_basename}] Error in Gemini 2.5 Pro: {e_pro}", exc_info=True)
                additional_text = "\n_Gemini 2.5 Pro:_ Failed.\n" + USERNAME

        return timestamp + analysis_result + additional_text

    except Exception as e_analysis:
        logger.error(f"[{file_basename}] Video analysis failed: {e_analysis}", exc_info=True)
        return timestamp + "Video analysis failed."

    finally:
        if video_file and hasattr(video_file, 'name'):
            try:
                logger.info(f"[{file_basename}] Deleting Gemini file {video_file.name}...")
                genai.delete_file(video_file.name)
                logger.info(f"[{file_basename}] Gemini file {video_file.name} deleted.")
            except Exception as del_e:
                # Log as warning, failure to delete isn't critical for main flow
                logger.warning(f"[{file_basename}] Failed to delete Gemini file {video_file.name}: {del_e}", exc_info=True)
                pass

# --- FileHandler (uses executor) ---
class FileHandler(FileSystemEventHandler):
    def __init__(self, loop, app):
        self.loop = loop
        self.app = app
        # Use the root logger configured globally
        self.logger = logging.getLogger(__name__) # Get logger specific to this class if needed

    def on_created(self, event):
        # Ignore directory creation events
        if event.is_directory:
            return
        # Filter for .mp4 files early
        if not event.src_path.endswith('.mp4'):
            # Log at DEBUG level if you want to see non-mp4 files being ignored
            # self.logger.debug(f"Ignoring non-mp4 file: {event.src_path}")
            return

        coro = self.handle_event(event)
        # Ensure the loop is running before scheduling
        if self.loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self.loop)
        else:
            self.logger.warning(f"Event loop not running when trying to schedule handler for {event.src_path}")

    async def handle_event(self, event):
        file_path = event.src_path
        file_basename = os.path.basename(file_path)
        self.logger.info(f"[{file_basename}] New file detected: {file_path}")

        # Add check for file stability/size before proceeding
        try:
            await self.wait_for_file_stable(file_path, file_basename)
        except FileNotFoundError:
            self.logger.warning(f"[{file_basename}] File disappeared before analysis could start: {file_path}")
            return
        except Exception as e_wait:
            self.logger.error(f"[{file_basename}] Error waiting for file stability: {e_wait}", exc_info=True)
            return # Don't proceed if stability check fails

        try:
            current_loop = asyncio.get_running_loop()
            # Use the globally configured logger within analyze_video
            video_response = await current_loop.run_in_executor(
                executor, analyze_video, file_path
            )
            self.logger.info(f"[{file_basename}] Analysis complete.")
        except Exception as e:
            self.logger.error(f"[{file_basename}] Error running analyze_video in executor: {e}", exc_info=True)
            video_response = f"_{file_basename[:6]}:_ Failed to analyze video."

        # --- Telegram Sending Logic ---
        try:
            self.logger.info(f"[{file_basename}] Sending message with button to Telegram...")
            # Ensure VIDEO_FOLDER ends with a separator for clean path joining/stripping
            safe_video_folder = os.path.join(VIDEO_FOLDER, '') # Adds separator if missing

            if file_path.startswith(safe_video_folder):
                 callback_file = file_path[len(safe_video_folder):]
                 # Normalize separators for callback data (e.g., replace \ with /)
                 callback_file = callback_file.replace(os.path.sep, '/')
            else:
                 self.logger.warning(f"[{file_basename}] File path '{file_path}' does not start with VIDEO_FOLDER '{safe_video_folder}'. Using basename for callback.")
                 callback_file = file_basename # Fallback

            keyboard = [[InlineKeyboardButton("Глянути", callback_data=callback_file)]]
            reply_markup = InlineKeyboardMarkup(keyboard)

            await self.app.bot.send_message(
                chat_id=CHAT_ID, text=video_response, reply_markup=reply_markup, parse_mode='Markdown'
            )
            self.logger.info(f"[{file_basename}] Message with button sent successfully.")

        except Exception as e:
            # Log the specific error during Telegram sending
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

async def main():
    """Run bot and watcher, handle graceful shutdown."""
    stop_event = asyncio.Event()

    # Use task names for better debugging if needed
    telegram_task = asyncio.create_task(run_telegram_bot(stop_event), name="TelegramBotTask")
    watcher_task = asyncio.create_task(run_file_watcher(stop_event), name="FileWatcherTask")

    tasks = {telegram_task, watcher_task}
    logger.info("Application started. Press Ctrl+C to exit.")

    # Monitor tasks
    try:
        # Wait for either task to complete (normally shouldn't happen unless error or stop)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Check which task finished and log potential errors
        for task in done:
            task_name = task.get_name()
            try:
                result = task.result() # Raises exception if task failed
                logger.warning(f"Task '{task_name}' completed unexpectedly without error. Result: {result}")
            except asyncio.CancelledError:
                 logger.info(f"Task '{task_name}' was cancelled.") # Expected during shutdown
            except Exception as task_exc:
                logger.error(f"Task '{task_name}' failed with exception:", exc_info=task_exc)

        # If any task finished (unexpectedly or due to error), signal shutdown
        if not stop_event.is_set():
             logger.warning("A task finished unexpectedly. Initiating shutdown...")
             stop_event.set()

        # Wait for remaining tasks if any (should be handled by finally block now)
        # if pending:
        #    await asyncio.wait(pending)

    except KeyboardInterrupt:
        logger.info("\nCtrl+C detected. Initiating graceful shutdown...")
        stop_event.set()

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
        logger.info("Main application finished cleanly.")


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