# Analyze Video Bot

This project is a Python-based application that monitors a folder for new video files, analyzes them using the Gemini AI platform, and sends the results to a Telegram chat. It is designed to work with surveillance camera footage and provides insights about the motion detected in the videos.

## Features

- **Folder Monitoring**: Automatically detects new `.mp4` files in a specified folder.
- **Video Analysis**: Uses Gemini AI to analyze the content of the video and generate a description of the motion detected.
- **Telegram Integration**: Sends the analysis results to a Telegram chat, either as a message with an inline button or by uploading the video.
- **Customizable Prompts**: Tailored prompts for specific use cases, such as identifying dogs or suspicious activity near a parked car.

## Requirements

- Python 3.10 or higher
- A Gemini AI API key
- A Telegram bot token
- The following Python libraries:
  - `python-telegram-bot`
  - `python-dotenv`
  - `watchdog`
  - `google-generativeai`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/analyze-video.git
   cd analyze-video
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project directory and add the following environment variables:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   TELEGRAM_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   TELEGRAM_NOTIFY_USERNAME=your_telegram_username
   VIDEO_FOLDER=/path/to/your/video/folder
   ```

## Usage

1. Start the application:
   ```bash
   python main.py
   ```

2. Place `.mp4` video files in the folder specified in the `VIDEO_FOLDER` environment variable. The application will automatically detect and process them.

3. Check the Telegram chat for the analysis results.

## How It Works

1. **File Monitoring**: The application uses the `watchdog` library to monitor the specified folder for new `.mp4` files.
2. **Video Analysis**: When a new video is detected, it is uploaded to Gemini AI for analysis. The AI generates a description of the motion detected in the video.
3. **Telegram Notification**: The analysis results are sent to a Telegram chat
4. **Callback Handling**: When a button is clicked, the button_callback function sends the corresponding video as a reply to the message with the button.