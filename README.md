# Analyze Video Bot

This project is a Python-based application that monitors a folder for new video files, analyzes them for motion and objects, generates descriptions with the Gemini AI platform, and sends results to a Telegram chat. It's optimized for surveillance footage, featuring intelligent object detection with YOLOv11, gate crossing alerts, optional Tesla integration, dynamic AI model selection, and robust performance management.

---

## Features

- **Folder Monitoring**: Automatically processes new `.mp4` files in a specified folder using `watchdog`.
- **Intelligent Video Analysis (OpenCV & YOLOv11)**:
  - **Object Detection & Tracking**: Uses a YOLOv11 model (optimized with OpenVINO) to detect and track objects like people and cars, assigning them stable IDs. The model has been pre-trained on footage from a wide-angle camera located on a second floor, enabling it to accurately identify distorted objects.
  - **Gate Crossing Detection**: Identifies and sends a special notification when a person crosses a predefined horizontal line in the frame, indicating entry or exit.
  - **Cropped ROI Analysis**: Performs analysis on a smaller, padded region around the ROI for better performance.
  - **Smart Event Filtering**: Differentiates between significant, insignificant, and noisy motion events based on duration.
  - **Highlight Clips**: Generates clips for significant motion with tracked objects and bounding boxes. Long events are automatically sped up (2x).
  - **Insignificant Motion Snapshots**: Extracts and sends a single frame for brief motion events, now with object detection boxes drawn on them.
- **Tesla Integration (Optional)**:
  - **State of Charge (SoC) Display**: If a car is detected in a predefined location, the bot fetches the Tesla's SoC and displays it directly on the video highlight clip.
  - **Efficient Caching**: Caches the SoC in `tesla_soc.txt` and only queries the API periodically or when the cache is stale to avoid waking the vehicle unnecessarily.
- **Dynamic Gemini AI Analysis**:
  - **Time-Based Model Selection**: Automatically switches between different Gemini models (e.g., Pro vs. Flash) based on the time of day for cost optimization.
  - **Fallback Models**: Includes logic to fall back to secondary and final models if the primary one fails.
  - **Custom Prompts**: Uses a `prompt.txt` file for tailored analysis.
- **Robust Telegram Integration**:
  - **Grouped Notifications**: Combines multiple insignificant/no-motion events into a single, editable Telegram message to reduce clutter.
  - **Interactive Callbacks**: Allows users to request the full original video via inline buttons.
  - **Media Handling**: Sends highlight clips as animations and insignificant motion as photos.
- **Performance & Stability**:
  - **Dual-Executor Design**: Uses separate, single-worker thread pools for CPU-bound (video analysis) and I/O-bound (API calls) tasks to prevent system overload.
  - **Graceful Shutdown & Auto-Restart**: Automatically restarts the script if any of the Python files is modified, with robust shutdown logic.
  - **Battery Monitoring**: Appends battery status to notifications if the device is on battery power and running low (requires `psutil`).
  - **Low Hardware Requirements**: Optimized to be efficient without losing accuracy, tested on Intel Core m5 CPU with 8Gb of RAM.
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
  - `ultralytics`
  - `openvino`
  - `teslapy` (optional, for Tesla integration)

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
   # --- Core ---
   GEMINI_API_KEY=your_gemini_api_key
   TELEGRAM_TOKEN=your_telegram_bot_token
   TELEGRAM_CHAT_ID=your_telegram_chat_id
   TELEGRAM_NOTIFY_USERNAME=your_telegram_username
   VIDEO_FOLDER=/path/to/your/video/folder
   LOG_PATH=logs/

   # --- Object Detection ---
   # Path to the exported OpenVINO model directory
   OBJECT_DETECTION_MODEL_PATH=best_openvino_model

   # --- Tesla Integration (Optional) ---
   TESLA_EMAIL=your_tesla_account_email
   TESLA_REFRESH_TOKEN=your_tesla_api_refresh_token
   ```

5. **Configure Regions of Interest (ROI):**
   - Generate a `roi.json` file defining the coordinates of the area you want to monitor for initial motion.
   - You can use the script in `tools/gate_motion_detector.py` as a starting point to select an ROI for your video.

6. **(Optional) Configure AI Models and Prompts:**
    - Place your custom analysis instructions in `prompt.txt`.
    - Configure Gemini model selection via `gemini_models.env` (read on every analysis call, no restart needed):
       ```env
       MODEL_PRO=
       MODEL_MAIN=gemini-2.5-flash
       MODEL_FALLBACK=gemini-2.5-flash-lite
       MODEL_FINAL_FALLBACK=
       ```
       - If `MODEL_PRO` is set and the current hour is between 09 and 13, `MODEL_PRO` is used as the main model and `MODEL_MAIN` becomes the fallback.
       - If `MODEL_PRO` is empty or outside 09–13, `MODEL_MAIN` is used as main and `MODEL_FALLBACK` as fallback.
       - If `MODEL_FINAL_FALLBACK` is set, it will be used as a last resort.
       - Known codenames are displayed with responses: `gemini-3-flash-preview → 3FP`, `gemini-2.5-flash → 2.5F`, `gemini-2.5-flash-lite → 2.5FL`, `gemini-2.5-pro → 2.5P` (unknown models display `FF` for final fallback).

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
   - **Video Analysis (CPU-Bound Task):** The video is passed to the `motion_executor`.
     - **Motion Detection:** OpenCV analyzes frames within a cropped ROI to find initial motion, filtering out noise.
     - **Object Tracking:** If significant motion is found, the `ultralytics` YOLO model tracks objects (people, cars) across frames.
     - **Event Classification:** The script determines the event type: `gate_crossing`, `significant_motion`, `insignificant_motion`, or `no_motion`.
     - **Artifact Generation:** A highlight clip (.mp4) or insignificant motion snapshots (.jpg) are created in the `temp/` directory.
   - **AI Analysis (I/O-Bound Task):** The result is passed to the `io_executor`.
     - For gate crossings or off-peak hours, Gemini is skipped.
     - For significant motion during peak hours, the highlight clip is sent to Gemini for a text description.

3. **Telegram Notification:**
   - A `telegram_lock` ensures that messages are sent or edited one at a time.
   - **Gate Crossing:** A special, high-priority message is sent immediately.
   - **Significant Motion:** A message is sent with the generated highlight clip and the AI description.
   - **Insignificant/No Motion:** Events are grouped into a single, editable message to avoid spam.

4. **Callback Handling:**
   - When a button is clicked, the bot retrieves the corresponding full video file and sends it as a reply.

5. **Self-Monitoring & Auto-Restart:**
   - A separate `watchdog` instance monitors `*.py`. If any Python file is modified, it triggers a graceful shutdown and restarts the script.

---

## Customization

- **Prompt:** Edit `prompt.txt` to change the analysis prompt sent to Gemini AI.
- **AI Models:** Edit `gemini_models.env` to control which Gemini models are used; changes apply without restarting the app.
- **Motion & Detection Parameters:** Adjust constants in `main.py` (e.g., `MIN_CONTOUR_AREA`, `CONF_THRESHOLD`, `LINE_Y`) to fine-tune sensitivity.
- **ROI:** Modify `roi.json` to change the monitored area.
- **Object Detection Model:** Replace the contents of the `best_openvino_model` directory with your own exported YOLOv11 OpenVINO model.
  - Follow instructions in [tools/finetuning/](tools/finetuning/) to train your own model if needed.

---

## Log Dashboard

A lightweight web dashboard that reads existing log files and provides per-day insights and a readable log viewer.

- Per-day views: timestamp, severity, video basename, message
- Filters: Severity and Status (`no_motion`, `gate_crossing`, `no_significant_motion`, `error`)
- Status Counts with totals and filtered counts
- Collapsible Per-Video Summary: Start time (first “New file detected”), status, raw events, processing time (prefers motion detection; falls back to full processing)
- Processing Times Chart: evenly spaced bars, colored by status, with min/avg/max guide lines and right-aligned axis labels; bars are clickable to jump to logs
- Status-based colors everywhere: filename colors in logs match the chart’s status families with deterministic shade variations; status badges use the same colors
- Quick access: `/today` route renders the latest day directly
- Responsive UX: stacked log entries on mobile and floating up/down buttons (visible on mobile and desktop)
- “Available Days” page shows today first, then past days in descending order

### Video Playback

- Embedded player: Clicking a video name in Per-Video Summary opens an embedded HTML5 player on the page.
- Timestamp links: Clicking the start time jumps to the first log entry for that video.
- Preserves filters: Opening a video keeps current filters (severity/status/gate) unchanged.
- Gate indicator: An arrow (↑/↓/↕) appears in the Status column indicating gate direction (both counts as up and down).
- Requirements:
   - Set `VIDEO_FOLDER` in your environment to the root folder where original `.mp4` files reside (monitored recursively).
   - The dashboard serves files via `GET /video/{basename}`. Only `.mp4` basenames are allowed; the first matching file is returned.
- Route details:
   - `play=<basename>` query parameter triggers the embedded player (e.g., `/today?play=YYYYMMDD_HHMMSS.mp4#player`).
   - The Per-Video Summary remains open when a player is shown and highlights the currently opened video row.

### Enable from main.py

Set in `.env`:

```bash
ENABLE_LOG_DASHBOARD=true
LOG_DASHBOARD_PORT=8000        # optional, defaults to 8000
LOG_DASHBOARD_HOST=0.0.0.0     # optional, defaults to 0.0.0.0
```

With `ENABLE_LOG_DASHBOARD=true`, the dashboard starts in a background thread and stops automatically on graceful shutdown. It will be reachable at `http://<host>:<port>` on your LAN.

Auto-reload:
- When the dashboard is enabled from `main.py`, changes in `tools/log_dashboard/` (Python, HTML, CSS) are detected and the dashboard restarts automatically.

### Standalone run

You can also run it independently:

```bash
python -m uvicorn tools.log_dashboard.app:app --port 8000 --reload
```

Optional configuration:

- `LOG_PATH`: directory containing `video_processor.log` and rotated files
- `LOG_BASENAME`: default `video_processor.log`

See [tools/log_dashboard/README.md](tools/log_dashboard/README.md) for details.
