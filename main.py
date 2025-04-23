import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from telegram import Bot, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CallbackQueryHandler
import datetime
import asyncio
from random import randint

from watchdog.events import FileSystemEventHandler
from watchdog.observers.polling import PollingObserver as Observer

load_dotenv()  ## load all the environment variables

api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=api_key)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
USERNAME = os.getenv("TELEGRAM_NOTIFY_USERNAME")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER")
telegram_bot = Bot(token=TELEGRAM_TOKEN)

# Initialize the Application with a larger connection pool size and timeout
application = Application.builder() \
    .token(TELEGRAM_TOKEN) \
    .http_version("1.1") \
    .connection_pool_size(32) \
    .build()

class FileHandler(FileSystemEventHandler):
    def __init__(self, loop):
        self.loop = loop  # Pass the main asyncio event loop

    def on_created(self, event):  # Synchronous wrapper
        asyncio.run_coroutine_threadsafe(self.handle_event(event), self.loop)

    async def handle_event(self, event):  # Asynchronous method
        await asyncio.sleep(randint(10, 15))  # Wait for the file to be fully written
        print(f"New file detected: {event.src_path}")
        if event.src_path.endswith('.mp4'):
            file_path = event.src_path
            video_response = analyze_video(file_path)
            
            now = datetime.datetime.now()

            if False: #("Отакої!" in video_response) and (9 <= now.hour <= 14):
                print("Sending video to Telegram...")
                try:
                    with open(file_path, 'rb') as video_file:
                        await telegram_bot.send_video(
                            chat_id=CHAT_ID,
                            video=video_file,
                            caption=video_response,
                            parse_mode='Markdown'
                        )
                except Exception as e:
                    print(f"Failed to send video to Telegram: {e}")
            else:
                print("Sending message with button to Telegram...")
                try:
                    callback_file = file_path.removeprefix(VIDEO_FOLDER)

                    # Create an inline button
                    keyboard = [
                        [InlineKeyboardButton("Глянути", callback_data=callback_file)]
                    ]
                    reply_markup = InlineKeyboardMarkup(keyboard)

                    await asyncio.sleep(randint(0, 9))  # Random delay before sending the message
                    # Send the message with the button
                    await telegram_bot.send_message(
                        chat_id=CHAT_ID,
                        text=video_response,
                        reply_markup=reply_markup,
                        parse_mode='Markdown'
                    )
                except Exception as e:
                    print(f"Failed to send message with button to Telegram: {e}")
            print(f"Done processing: {file_path}")
        else:
            print("Not a video file, skipping...")

def analyze_video(video_path):
    """Extract insights from the video using Gemini Flash."""
    timestamp = f"_{os.path.basename(video_path)[:6]}:_ "
    try:
        video_file = genai.upload_file(path=video_path)
    except:
        return timestamp + "Video upload failed."
    print(f"Completed upload: {video_file.uri}")

    while video_file.state.name == "PROCESSING":
        time.sleep(10)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name == "FAILED":
        return timestamp + "Video processing failed."
    
    prompt = """This video is from a surveillance camera, located on the second floor above the entrance to the house.
      The camera is facing the street. Camera is set to record only when there is motion. Audio is unrelated and should be ignored.
      Describe in one sentence what motion triggered recording. Don't mention that this triggered recording, just describe the action.
      If any dog is present in the video, describe it and the person, include keyword "*Отакої!*" and state confidence level.
      If a dog is not present in the video, ensure that the phrase "*Отакої!*" is NOT included in the response.
      If someone is doing something to the parked red Tesla (not just passing by), describe what they are doing and include keyword "Хм...".
      Only return the output and nothing else. Respond in Ukrainian.
      """

    model = genai.GenerativeModel(model_name="models/gemini-2.0-flash")

    try:
        response = model.generate_content([prompt, video_file],
                                        request_options={"timeout": 600})
    except:
        genai.delete_file(video_file.name)
        return timestamp + "Video analysis failed."

    print(response.text)
    
    additional_text = ""
    now = datetime.datetime.now()
    if ("Отакої!" in response.text) and (9 <= now.hour <= 15):
        model = genai.GenerativeModel(model_name="models/gemini-2.5-pro-exp-03-25")
        print("Trying Gemini 2.5 Pro for better results.")
        try:
            response_new = model.generate_content([prompt, video_file],
                                                request_options={"timeout": 600})
            print(response_new.text)
            additional_text = "\n_Gemini 2.5:_ " + response_new.text + "\n" + USERNAME
        except Exception as e:
            print(f"Error in Gemini 2.5 Pro: {e}")
            additional_text = "\n_Gemini 2.5:_ Failed.\n" + USERNAME

    try:
        genai.delete_file(video_file.name)
    except: 
        pass
    return timestamp + response.text + additional_text

# Callback handler for the button click
async def button_callback(update, context):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback (await is required)

    # Get the callback data and map it to the file path
    callback_file = query.data
    print(f"Callback data received: {callback_file}")  # Debug print
    file_path = VIDEO_FOLDER + callback_file

    if not file_path:
        print(f"Invalid callback data: {file_path}")
        return

    print(f"Button clicked, sending video: {file_path}")

    try:
        with open(file_path, 'rb') as video_file:
            await context.bot.send_video(  # Await the async method
                chat_id=query.message.chat_id,
                video=video_file,
                parse_mode='Markdown',
                caption="Осьо відео",
                reply_to_message_id=query.message.message_id  # Reply to the message with the button
            )
    except Exception as e:
        print(f"Failed to send video: {e}")

# Add the callback handler to the application
application.add_handler(CallbackQueryHandler(button_callback))

async def run_telegram_bot():
    """Run the Telegram bot."""
    print("Starting Telegram bot...")
    await application.initialize()  # Initialize the application
    await application.start()       # Start the application
    await application.updater.start_polling()  # Start polling for updates
    #await application.stop()        # Stop the application when done

async def run_file_watcher():
    """Run the file-watching logic."""
    print("Watching for new files...")
    loop = asyncio.get_event_loop()  # Get the main asyncio event loop
    event_handler = FileHandler(loop)  # Pass the loop to the handler
    observer = Observer()
    observer.schedule(event_handler, path=VIDEO_FOLDER, recursive=True)
    observer.start()

    try:
        while True:
            await asyncio.sleep(1)  # Keep the loop running
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

async def main():
    """Run both the Telegram bot and file watcher concurrently."""
    await asyncio.gather(
        run_telegram_bot(),
        run_file_watcher()
    )

# Run the main function using the existing event loop
if __name__ == "__main__":
    asyncio.run(main())