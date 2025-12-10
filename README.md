# Analyze Video Bot

This project is a Python-based application that monitors a folder for new video files, analyzes them with OpenCV for motion in specified area, generates descriptions with the Gemini AI platform, and sends results to a Telegram chat. It's optimized for surveillance footage, featuring intelligent motion detection, dynamic AI model selection, and robust performance management.

---

## Features

- **Folder Monitoring**: Automatically processes new `.mp4` files in a specified folder using `watchdog`.
- **Advanced Motion Detection**:
  - **Cropped ROI Analysis**: Performs analysis on a smaller, padded region around the ROI for better performance.
  - **Background Pre-training**: Stabilizes the background model by training it on a later segment of the video, improving detection accuracy from the start.
  - **Smart Event Filtering**: Differentiates between significant, insignificant, and noisy motion events based on duration and size.
  - **Highlight Clips**: Generates clips for significant motion with interpolated bounding boxes for smooth tracking. Long events are automatically sped up.
  - **Insignificant Motion Snapshots**: Extracts and sends a single frame for brief, insignificant motion events instead of discarding them.
- **Dynamic Gemini AI Analysis**:
  - **Time-Based Model Selection**: Automatically switches between different Gemini models (e.g., Pro vs. Flash) based on the time of day for cost and performance optimization.
  - **Fallback Models**: Includes logic to fall back to secondary and final models if the primary one fails.
  - **Custom Prompts**: Uses a `prompt.txt` file for tailored analysis.
- **Robust Telegram Integration**:
  - **Grouped Notifications**: Combines multiple insignificant/no-motion events into a single, editable Telegram message to reduce clutter.
  - **Interactive Callbacks**: Allows users to request the full original video via inline buttons.
  - **Media Handling**: Sends highlight clips as animations and insignificant motion as photos.
- **Performance & Stability**:
  - **Dual-Executor Design**: Uses separate, single-worker thread pools for CPU-bound (motion detection) and I/O-bound (AI analysis) tasks to prevent system overload and ensure sequential processing.
  - **Graceful Shutdown & Auto-Restart**: Automatically restarts the script if `main.py` is modified, with robust shutdown logic.
  - **Battery Monitoring**: Appends battery status to notifications if the device is on battery power and running low (requires `psutil`).
- **Enhanced Logging**:
  - **Custom Log Rotation**: Creates daily rotating log files with a clear `YYYY-MM-DD` naming convention.
  - **Network Error Filtering**: Suppresses noisy network-related stack traces to keep logs clean.

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
  - `psutil`

Install dependencies with:
```bash
pip install -r requirements.txt
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alivespirit/analyze-video.git
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

5. **Configure Regions of Interest (ROI):**
   - Generate a `roi.json` file defining the coordinates of the area you want to monitor.
   - You can use the script in `tools/gate_motion_detector.py` as a starting point to select an ROI for your video.

6. **(Optional) Configure AI Models and Prompts:**
   - Place your custom analysis instructions in `prompt.txt`.
   - To enable dynamic model switching, create the following files in the project root:
     - `model_pro`: Contains the name of your preferred high-accuracy model (e.g., `gemini-2.5-pro`). This model will be used during peak hours (9 AM - 1 PM), with a fallback to default `gemini-2.5-flash`.
     - `model_final_fallback`: Contains the name of a last-resort model to be used if all others fail.

---

## Usage

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Place `.mp4` video files in the folder specified by the `VIDEO_FOLDER` environment variable.**  
   The application will automatically detect, process, and send a notification to your Telegram chat.

3. **Interact with the bot in Telegram.**  
   - View highlight clips and insignificant motion snapshots.
   - Click the "Глянути" (View) or timestamp buttons to receive the full original video.

---

## How It Works

1. **File Monitoring:**  
   - `watchdog` recursively monitors the `VIDEO_FOLDER`. When a new `.mp4` file is detected, it waits for the file size to stabilize before queuing it for analysis.

2. **Processing Pipeline:**  
   - **Motion Detection (CPU-Bound Task):** The video is passed to the `motion_executor`. OpenCV analyzes the frames within the cropped ROI, using a pre-trained background model to identify and classify motion events. It generates a highlight clip, insignificant motion snapshots, or determines there was no significant motion.
   - **AI Analysis (I/O-Bound Task):** The result (highlight clip or full video) is passed to the `io_executor`. The script selects a Gemini model based on the time of day and file-based configuration, then sends the video for analysis.

3. **Telegram Notification:**  
   - A `telegram_lock` ensures that messages are sent or edited one at a time, preventing race conditions.
   - **Significant Motion:** A message is sent with the generated highlight clip (as an animation) and a button to request the full video.
   - **Insignificant/No Motion:** Events are grouped into a single message. The message text is updated, and new buttons are added for each subsequent event (up to 4).

4. **Callback Handling:**  
   - When a button is clicked, the bot retrieves the corresponding full video file and sends it as a reply.

5. **Self-Monitoring & Auto-Restart:**  
   - A separate `watchdog` instance monitors `main.py`. If the file is modified, it triggers a graceful shutdown of all tasks and restarts the script using `os.execv`.

---

## Customization

- **Prompt:** Edit `prompt.txt` to change the analysis prompt sent to Gemini AI.
- **AI Models:** Create or remove `model_pro` and `model_final_fallback` to control which Gemini models are used.
- **Motion Detection Parameters:** Adjust constants in `main.py` (e.g., `MIN_CONTOUR_AREA`, `MAX_EVENT_GAP_SECONDS`) to fine-tune motion sensitivity.
- **Logging:** Change `LOG_PATH` and log rotation settings in `main.py` as needed.
- **ROI:** Modify `roi.json` to change the monitored area in the video frame.
