import os
import time
import datetime
import asyncio
from random import randint
import concurrent.futures # Needed for executor

from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CallbackQueryHandler
import google.generativeai as genai

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

load_dotenv()  ## load all the environment variables

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
USERNAME = os.getenv("TELEGRAM_NOTIFY_USERNAME")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize the Application
application = Application.builder() \
    .token(TELEGRAM_TOKEN) \
    .http_version("1.1") \
    .get_updates_http_version("1.1") \
    .connection_pool_size(32) \
    .pool_timeout(60) \
    .build()

# Thread Pool Executor for Blocking Tasks
executor = concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4)

# --- analyze_video function (remains synchronous, uses original models) ---
def analyze_video(video_path):
    """Extract insights from the video using Gemini. Runs in executor."""
    print(f"[{os.path.basename(video_path)}] Starting analysis...")
    timestamp = f"_{os.path.basename(video_path)[:6]}:_ "
    video_file = None
    try:
        video_file = genai.upload_file(path=video_path)
    except Exception as e:
        print(f"[{os.path.basename(video_path)}] Video upload failed: {e}")
        return timestamp + "Video upload failed."

    print(f"[{os.path.basename(video_path)}] Completed upload: {video_file.uri}")

    try:
        while video_file.state.name == "PROCESSING":
            print(f"[{os.path.basename(video_path)}] Waiting for processing...")
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            print(f"[{os.path.basename(video_path)}] Video processing failed state.")
            return timestamp + "Video processing failed."

        prompt = """This video is from a surveillance camera, located on the second floor above the entrance to the house.
        The camera is facing the street. Camera is set to record only when there is motion. Audio is unrelated and should be ignored.
        Describe in one sentence what motion triggered recording. Don't mention that this triggered recording, just describe the action.
        If any dog is present in the video, describe it and the person, include keyword "*Отакої!*" and state confidence level.
        If a dog is not present in the video, ensure that the phrase "*Отакої!*" is NOT included in the response.
        If someone is doing something to the parked red Tesla (not just passing by), describe what they are doing and include keyword "Хм...".
        Only return the output and nothing else. Respond in Ukrainian.
        """

        model_flash = genai.GenerativeModel(model_name="models/gemini-2.0-flash")
        analysis_result = ""
        additional_text = ""

        print(f"[{os.path.basename(video_path)}] Generating content (Flash 2.0)...")
        response = model_flash.generate_content([prompt, video_file],
                                                request_options={"timeout": 600})
        analysis_result = response.text
        print(f"[{os.path.basename(video_path)}] Gemini Flash 2.0 response received.")

        now = datetime.datetime.now()
        if ("Отакої!" in analysis_result) and (9 <= now.hour <= 15):
            model_pro = genai.GenerativeModel(model_name="models/gemini-2.5-pro-exp-03-25")
            print(f"[{os.path.basename(video_path)}] Trying Gemini 2.5 Pro...")
            try:
                response_new = model_pro.generate_content([prompt, video_file],
                                                          request_options={"timeout": 600})
                print(f"[{os.path.basename(video_path)}] Gemini Pro 2.5 response received.")
                additional_text = "\n_Gemini 2.5:_ " + response_new.text + "\n" + USERNAME
            except Exception as e_pro:
                print(f"[{os.path.basename(video_path)}] Error in Gemini 2.5 Pro: {e_pro}")
                additional_text = "\n_Gemini 2.5:_ Failed.\n" + USERNAME

        return timestamp + analysis_result + additional_text

    except Exception as e_analysis:
        print(f"[{os.path.basename(video_path)}] Video analysis failed: {e_analysis}")
        return timestamp + "Video analysis failed."

    finally:
        if video_file and hasattr(video_file, 'name'):
            try:
                print(f"[{os.path.basename(video_path)}] Deleting Gemini file {video_file.name}...")
                genai.delete_file(video_file.name)
                print(f"[{os.path.basename(video_path)}] Gemini file deleted.")
            except Exception as del_e:
                print(f"[{os.path.basename(video_path)}] Failed to delete Gemini file {video_file.name}: {del_e}")

# --- FileHandler (uses executor) ---
class FileHandler(FileSystemEventHandler):
    def __init__(self, loop, app):
        self.loop = loop
        self.app = app

    def on_created(self, event):
        coro = self.handle_event(event)
        asyncio.run_coroutine_threadsafe(coro, self.loop)

    async def handle_event(self, event):
        if not event.src_path.endswith('.mp4'):
            return

        file_path = event.src_path
        file_basename = os.path.basename(file_path)
        print(f"[{file_basename}] New file detected: {file_path}")

        await asyncio.sleep(randint(10, 15))
        print(f"[{file_basename}] Finished waiting, starting analysis via executor...")

        try:
            current_loop = asyncio.get_running_loop()
            video_response = await current_loop.run_in_executor(
                executor, analyze_video, file_path
            )
            print(f"[{file_basename}] Analysis complete.")
        except Exception as e:
            print(f"[{file_basename}] Error running analyze_video in executor: {e}")
            video_response = f"_{file_basename[:6]}:_ Failed to analyze video due to system error."

        try:
            if False: # Original logic for direct video send
                print(f"[{file_basename}] Sending video to Telegram...")
                try:
                    with open(file_path, 'rb') as video_file:
                        await self.app.bot.send_video(
                            chat_id=CHAT_ID, video=video_file, caption=video_response, parse_mode='Markdown'
                        )
                except FileNotFoundError:
                    print(f"[{file_basename}] File not found when trying to send video: {file_path}")
                except Exception as e:
                    print(f"[{file_basename}] Failed to send video to Telegram: {e}")
            else:
                print(f"[{file_basename}] Sending message with button to Telegram...")
                try:
                    safe_video_folder = VIDEO_FOLDER if VIDEO_FOLDER.endswith(os.path.sep) else VIDEO_FOLDER + os.path.sep
                    if file_path.startswith(safe_video_folder):
                         callback_file = file_path[len(safe_video_folder):]
                    else:
                         print(f"[{file_basename}] WARNING: File path does not start with VIDEO_FOLDER.")
                         callback_file = file_basename

                    keyboard = [[InlineKeyboardButton("Глянути", callback_data=callback_file)]]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await asyncio.sleep(randint(0, 5))

                    await self.app.bot.send_message(
                        chat_id=CHAT_ID, text=video_response, reply_markup=reply_markup, parse_mode='Markdown'
                    )
                except Exception as e:
                    print(f"[{file_basename}] Failed to send message with button to Telegram: {e}")

        except Exception as e:
            print(f"[{file_basename}] Error during Telegram sending block: {e}")
        finally:
             print(f"[{file_basename}] Telegram interaction finished.")


# --- Callback Handler ---
async def button_callback(update, context):
    query = update.callback_query
    await query.answer()

    callback_file = query.data
    file_path = os.path.join(VIDEO_FOLDER, callback_file)
    file_basename = os.path.basename(file_path)

    print(f"[{file_basename}] Button callback received for: {callback_file}")

    if not os.path.exists(file_path):
        print(f"[{file_basename}] Video file not found for callback: {file_path}")
        try:
            await query.edit_message_text(text=f"{query.message.text}\n\n_Відео файл не знайдено._", parse_mode='Markdown')
        except Exception as edit_e:
            print(f"[{file_basename}] Error editing message for not found file: {edit_e}")
        return

    print(f"[{file_basename}] Sending video from callback...")
    try:
        with open(file_path, 'rb') as video_file:
            await context.bot.send_video(
                chat_id=query.message.chat_id, video=video_file, parse_mode='Markdown',
                caption="Осьо відео", reply_to_message_id=query.message.message_id
            )
        print(f"[{file_basename}] Video sent successfully from callback.")
    except FileNotFoundError:
         print(f"[{file_basename}] Video file disappeared before sending from callback: {file_path}")
         try: await query.edit_message_text(text=f"{query.message.text}\n\n_Помилка: Відео файл зник._", parse_mode='Markdown')
         except Exception: pass
    except Exception as e:
        print(f"[{file_basename}] Failed to send video from callback: {e}")
        try: await query.edit_message_text(text=f"{query.message.text}\n\n_Помилка відправки відео._", parse_mode='Markdown')
        except Exception: pass

# Add the callback handler
application.add_handler(CallbackQueryHandler(button_callback))


# --- Main Execution and Shutdown Logic ---

# FIX: Corrected shutdown logic inside finally block
async def run_telegram_bot(stop_event):
    """Run the Telegram bot until stop_event is set."""
    print("Starting Telegram bot polling...")
    try:
        await application.initialize()
        await application.start()
        await application.updater.start_polling(poll_interval=1.0, timeout=20)
        print("Telegram bot started.")
        await stop_event.wait()
    except asyncio.CancelledError:
        print("Telegram bot task cancelled.")
    except Exception as e:
        print(f"Error in Telegram bot task: {e}")
        stop_event.set() # Signal shutdown on error
    finally:
        print("Stopping Telegram bot...")
        # Stop the updater first if it exists
        if application.updater:
            await application.updater.stop() # No .is_running check needed
        # Then stop the application itself if it's running
        if application.running:
            await application.stop()
        print("Telegram bot stopped.")

# FIX: Changed recursive back to True
async def run_file_watcher(stop_event):
    """Run the file-watching logic until stop_event is set."""
    print("Starting file watcher...")
    observer = None
    try:
        loop = asyncio.get_running_loop()
        event_handler = FileHandler(loop, application)
        observer = Observer()
        if not os.path.isdir(VIDEO_FOLDER):
            print(f"ERROR: VIDEO_FOLDER '{VIDEO_FOLDER}' does not exist or is not a directory.")
            stop_event.set()
            return

        observer.schedule(event_handler, path=VIDEO_FOLDER, recursive=True) # WATCH RECURSIVELY
        observer.start()
        print(f"Watching for new files in: {VIDEO_FOLDER} (Recursive Mode)")

        while not stop_event.is_set() and observer.is_alive():
            await asyncio.sleep(1)

    except asyncio.CancelledError:
        print("File watcher task cancelled.")
    except Exception as e:
        print(f"Error in File watcher task: {e}")
        stop_event.set() # Signal shutdown on error
    finally:
        print("Stopping file watcher...")
        if observer and observer.is_alive():
            observer.stop()
            observer.join(timeout=5.0)
            if observer.is_alive():
                 print("Warning: Observer thread did not stop cleanly after 5 seconds.")
        print("File watcher stopped.")

async def main():
    """Run bot and watcher, handle graceful shutdown."""
    stop_event = asyncio.Event()
    telegram_task = asyncio.create_task(run_telegram_bot(stop_event), name="TelegramBotTask")
    watcher_task = asyncio.create_task(run_file_watcher(stop_event), name="FileWatcherTask")

    tasks = {telegram_task, watcher_task}
    print("Application started. Press Ctrl+C to exit.")

    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        if pending:
            print("A task finished unexpectedly. Initiating shutdown...")
            stop_event.set()
            await asyncio.wait(pending) # Wait for the remaining task(s)

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Initiating graceful shutdown...")
        stop_event.set()

    except asyncio.CancelledError:
        print("Main task cancelled.")
        stop_event.set()

    finally:
        print("Shutdown sequence started.")
        if not stop_event.is_set():
            stop_event.set()

        print("Waiting for tasks to finish...")
        await asyncio.gather(*tasks, return_exceptions=True)
        print("All application tasks have finished.")

        print("Shutting down thread pool executor (allowing current analysis to finish)...")
        executor.shutdown(wait=True)
        print("Executor shut down.")
        print("Main application finished cleanly.")


# Run the main function
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # We expect KeyboardInterrupt to be handled gracefully inside main()
        # This outer catch simply prevents the final traceback from asyncio.run()
        print("\nExiting application.") # Or just 'pass' if you don't want any extra message