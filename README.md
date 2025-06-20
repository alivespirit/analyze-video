# Analyze Video Bot

This project is a Python-based application that monitors a folder for new video files, analyzes them using the Gemini AI platform, and sends the results to a Telegram chat. It is designed to work with surveillance camera footage and provides insights about the motion detected in the videos.

## Features

- **Folder Monitoring**: Automatically detects new `.mp4` files in a specified folder.
- **Video Analysis**: Uses Gemini AI to analyze the content of the video and generate a description of the motion detected.
- **Frame Extraction**: Extracts frame at the timestamp specified by Gemini AI showcasing the detected action.
- **Telegram Integration**: Sends the analysis results to a Telegram chat, optionally including extracted frame or short clip, with an inline button to request full video.
- **Customizable Prompts**: Tailored prompts for specific use cases, such as identifying dogs or suspicious activity near a parked car.
- **Self monitoring**: Automatically restart process if `main.py` was changed (useful if it is running on the remote server).

## Requirements

- Python 3.10 or higher
- A Gemini AI API key
- A Telegram bot token
- The following Python libraries:
  - `python-telegram-bot`
  - `python-dotenv`
  - `watchdog`
  - `google-generativeai`
  - `opencv-python`
  - `moviepy`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/analyze-video.git
   cd analyze-video
   ```

2. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project directory and add the following environment variables:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   TELEGRAM_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   TELEGRAM_NOTIFY_USERNAME=your_telegram_username
   VIDEO_FOLDER=/path/to/your/video/folder
   LOG_PATH=logs/
   ```

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. Place `.mp4` video files in the folder specified in the `VIDEO_FOLDER` environment variable. The application will automatically detect and process them.

3. Check the Telegram chat for the analysis results.

## How It Works

1. **File Monitoring**: 
    - The script uses `watchdog` to monitor the folder specified in `VIDEO_FOLDER`.
    - When a new `.mp4` file is detected, it triggers the `handle_event` method.
2. **Video Analysis**: 
    - The `analyze_video` function uploads the video to the Gemini AI API and retrieves the analysis results.
    - If the analysis detects specific keywords (e.g., "Отакої!"), it uses a more advanced Gemini model for better results.
3. **Telegram Notification**: 
    - The script uses the `python-telegram-bot` library to send messages and videos to Telegram.
    - If the analysis detects specific keywords, it checks response for timestamps, extracts frame at the timestamp or videoclip at timerange, and attaches this frame or clip to the message.
    - Interactive buttons are created using `InlineKeyboardMarkup` and `InlineKeyboardButton`.
4. **Callback Handling**: 
    - When a button is clicked, the `button_callback` function sends the corresponding video as a reply to the message with the button.