# Analyze Video Bot

This project is a Python-based application that monitors a folder for new video files, analyzes them using the Gemini AI platform, and sends the results to a Telegram chat. It is designed for surveillance camera footage and provides insights about detected motion, including highlight clips and interactive Telegram features.

---

## Features

- **Folder Monitoring**: Automatically detects new `.mp4` files in a specified folder using `watchdog`.
- **Motion Detection & Highlight Clips**: Detects significant motion events within Region Of Interest (ROI) using OpenCV, generates highlight clips with bounding boxes, and filters out noise/short events.
- **Gemini AI Analysis**: Uses Gemini AI to analyze video content and generate a description of detected motion. Handles API rate limits using a semaphore and exponential backoff.
- **Frame & Clip Extraction**: Extracts and sends highlight clips (as GIF/MP4 animation) for significant motion, and can extract frames at specific timestamps.
- **Telegram Integration**: Sends analysis results to a Telegram chat, including highlight clips and an inline button to request the full video.
- **Grouped "No Motion" Messages**: Groups multiple "no motion" events into a single Telegram message with multiple buttons.
- **Customizable Prompts**: Uses a prompt file (`prompt.txt`) for tailored analysis.
- **Self-Monitoring & Auto-Restart**: Automatically restarts the process if `main.py` is changed (useful for remote servers).
- **Robust Logging**: Daily rotating log files with custom naming, and logs to both file and console.

---

## Requirements

- Python 3.10 or higher
- A Gemini AI API key
- A Telegram bot token
- The following Python libraries:
  - `python-telegram-bot`
  - `python-dotenv`
  - `watchdog`
  - `google-genai`
  - `moviepy`
  - `opencv-python`
  - `numpy`

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/analyze-video.git
   cd analyze-video
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create a `.env` file in the project directory and add the following environment variables:**
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   TELEGRAM_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   TELEGRAM_NOTIFY_USERNAME=your_telegram_username
   VIDEO_FOLDER=/path/to/your/video/folder
   LOG_PATH=logs/
   ```

5. **Generate `roi.json` for your specific ROI and place it in the project directory.**  
   You can ask Gemini for a sample script to select ROI based on your video, or use script in `tools/gate_motion_detector.py`.

6. **(Optional) Place your custom prompt in `prompt.txt` in the project directory.**

---

## Usage

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Place `.mp4` video files in the folder specified by the `VIDEO_FOLDER` environment variable.**  
   The application will automatically detect and process them.

3. **Check the Telegram chat for analysis results, highlight clips, and interactive buttons.**

---

## How It Works

1. **File Monitoring:**  
   - Uses `watchdog` to monitor the `VIDEO_FOLDER` for new `.mp4` files. When a new file is detected, it waits for the file to become stable before processing.

2. **Motion Detection & Highlight Generation:**  
   - Detects significant motion events using OpenCV.
   - Filters out short/noisy events.
   - Generates highlight clips with bounding boxes for significant motion.
   - If no significant motion is found, groups up to 4 "no motion" events in a single Telegram message.

3. **Gemini AI Analysis:**  
   - Analyzes the video or highlight clip using Gemini AI.
   - Uses a semaphore to ensure only one Gemini API call at a time (avoiding rate limit errors).
   - Retries with exponential backoff if rate limits are hit.
   - Uses a custom prompt from `prompt.txt`.

4. **Telegram Notification:**  
   - Sends highlight clips as animations for significant motion, with a button to request the full video.
   - Groups "no motion" events into a single message with multiple buttons.
   - Handles Markdown formatting and Telegram API errors robustly.

5. **Callback Handling:**  
   - When a button is clicked, sends the corresponding video as a reply to the message.

6. **Self-Monitoring & Auto-Restart:**  
   - Watches `main.py` for changes and restarts the process automatically if the script is updated.

7. **Logging:**  
   - Logs to both console and daily rotating log files (with custom naming).
   - Logs errors, warnings, and info for all major actions.

---

## Troubleshooting

- **Quota/Rate Limit Errors:**  
  The script uses a semaphore and exponential backoff to avoid Gemini API rate limits. If you still hit limits, reduce the number of concurrent videos or increase the backoff time.
- **Telegram Formatting Errors:**  
  The script retries with escaped Markdown if Telegram rejects a message due to formatting.
- **File Not Found:**  
  If a video file is deleted before it can be sent, the bot will notify you in the chat.

---

## Customization

- **Prompt:**  
  Edit `prompt.txt` to change the analysis prompt sent to Gemini AI.
- **Motion Detection Parameters:**  
  Adjust constants like `MIN_CONTOUR_AREA`, `MIN_EVENT_DURATION_SECONDS`, etc., in `main.py` to fine-tune motion detection.
- **Logging:**  
  Change `LOG_PATH` and log rotation settings in `main.py` as needed.
