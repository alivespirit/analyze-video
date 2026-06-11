# Analyze Video Bot - Споглядайко

This project is a Python-based application that monitors a folder for new video files, analyzes them for motion and objects, generates descriptions with the Gemini AI platform, and sends results to a Telegram chat. It's designed for surveillance footage, featuring object detection/tracking with a custom-trained YOLOv12 model (exported to OpenVINO), gate crossing alerts, optional Tesla integration, dynamic AI model selection, and restart recovery.

---

## Features

- **Folder Monitoring**: Automatically processes new `.mp4` files in a specified folder using `watchdog`.
- **Intelligent Video Analysis (OpenCV & YOLOv12)**:
   - **Cropped ROI Analysis**: Performs motion detection on a smaller, padded region around the ROI for better performance.
   - **Object Detection & Tracking**: Uses a custom-trained YOLOv12 model exported to OpenVINO (loaded via the Ultralytics `YOLO` wrapper) to detect and track objects like people and cars, assigning them stable IDs. The model has been trained on footage from a wide-angle camera located on a second floor, enabling it to more accurately identify distorted objects.
  - **Gate Crossing Detection**: Identifies and sends a special notification when a person crosses a predefined horizontal line in the frame, indicating entry or exit.
  - **False-Positive Suppression**:
    - **Car-below-line filter**: Cars whose bounding box extends below the configured gate line (`LINE_Y`) are automatically discarded, reducing impossible detections (e.g., cars appearing to drive underground or through the gate).
    - **Static person suppression**: Person detections are tracked for movement and confidence. Entities that remain nearly stationary (center drift below threshold) and have low average confidence are suppressed as likely false positives (e.g., snow piles, statues). Suppression is only triggered after a minimum number of updates to avoid premature filtering.
    - **Probation/trust mechanism**: New person entities start in probation and are drawn for visibility, but do not count toward ROI, crossing, or unique person stats until trusted. Trust is granted if the entity moves enough or accumulates sufficient confidence over time. This prevents static false positives from affecting event logic.
    - All suppression parameters are configurable in `detect_motion.py` (see "Customization" below).
  - **Smart Event Filtering**: Differentiates between significant, insignificant, and noisy motion events based on duration.
  - **Highlight Clips**: Generates clips for significant motions with tracked objects and bounding boxes. Uses a CRF-based H.264 writer (libx264, CRF=28, yuv420p, faststart) with the encoder preset configurable via `VIDEO_WRITER_PRESET` (default `faster`). For 1080p/4K sources, highlight output is written at 1080p. Clips are saved to daily subdirectories (`TEMP_DIR/YYYYMMDD/`) and optionally kept after Telegram send (`KEEP_HIGHLIGHTS_CLIPS=true`, default) for viewing in the Android dashboard.
  - **Gate-Crossing Crop Magnifier**: While a person is within the gate band, a live zoomed-in picture-in-picture of their current bounding box is drawn in the lower-right corner (above the event text), connected to the box by a two-line "funnel" callout. The PiP border and funnel lines match the box color (green normally, red in the line tolerance). Single primary person — the most-recent line crosser, so when people cross one-by-one the PiP hands off to whoever just crossed (falls back to nearest-to-line / largest box otherwise); configurable via `GATE_CROP_OVERLAY_*` (see Customization).
  - **Long-Event Speed-Up**: Long events are rendered faster by writing fewer frames (frame skipping) while keeping the output FPS unchanged.
  - **Insignificant Motion Snapshots**: Can extract a representative frame for brief motion events; sending snapshots to Telegram is optional.
  - **Approximate Car Speed Estimation**:
    - Adds `XX km/h` to car labels using tracked bbox-center motion over time.
    - Accounts for tracker/output stride by using frame-index deltas and source FPS.
    - Applies heuristic perspective and wide-angle edge correction, plus median+EMA smoothing to reduce jitter.
    - Draws a short trailing line behind moving cars for readability.
  - **Car SpeedTrap (Line-to-Line Measurement)**:
    - Computes near-precise car speed from frame count between two vertical X-lines with known real-world distance.
    - Supports both driving directions (`X1 -> X2` and `X2 -> X1`).
    - Shows the latest SpeedTrap value in the event overlay and highlights moving cars inside the trap zone.
  - **Low-Resolution Clips (Motion-Only Mode)**: For specific low-res sources (640×360 and 896×512), runs ROI motion detection and generates a highlight at the source resolution (no tracking/ReID).
- **Dynamic Gemini AI Analysis**:
  - **Time-Based Model Selection**: Automatically switches between different Gemini models (e.g., Pro vs. Flash) based on the time of day for cost optimization.
  - **Fallback Models**: Includes logic to fall back to secondary and final models if the primary one fails.
  - **Custom Prompts**: Uses a `config/prompt.txt` file for tailored analysis.
- **Local LLM Frame Analysis (Ollama, Optional)**:
  - **Offloads low-motion clips from Gemini**: `no_person` and `no_significant_motion` videos (typically wind, shadows, sun) are described by a local vision model running on the worker GPU instead of consuming the limited Gemini daily quota — which is then reserved for `significant_motion` clips.
  - **Runs at all hours**: No quota, so these clips get a real description even off-peak (where Gemini was previously skipped with a placeholder).
  - **Two operating modes**: a single vision model, or a two-stage pipeline (a fast vision model grounds the scene, then a dedicated language model rewrites it in clean, idiomatic Ukrainian).
  - **Saved event frames**: Analyzes the representative frames `detect_motion` already saves for each low-motion event (single- or multi-frame in one call).
  - **Graceful fallback**: If Ollama is unreachable, the pipeline falls back to the original placeholder messages — nothing breaks.
- **Robust Telegram Integration**:
  - **Grouped Notifications**: Combines no-motion events into a single, editable Telegram message to reduce clutter.
  - **Event-Frame Photos**: `no_person` / `no_significant_motion` events are sent as a single photo (the most detailed event frame) with the local-LLM description as the caption and a "Глянути" full-video button — one self-contained message. The frame is downscaled before upload to cut bandwidth; toggle with `SEND_INSIGNIFICANT_FRAMES` (off → the description is delivered as a grouped text message instead).
  - **Leftover-Frame Replies**: a `gate_crossing` / `significant_motion` video can also produce leftover low-motion frames (e.g. a second event where a poorly-tracked person appears). After the highlight is sent, the best such frame is analyzed by the local LLM (off the critical path) and posted as a threaded reply to the highlight — so an occasionally-interesting frame isn't dropped, without delaying the highlight. Toggle with `SEND_GATE_EXTRA_FRAMES` (default on).
  - **Interactive Callbacks**: Allows users to request the full original video via inline buttons.
  - **Media Handling**: Sends highlight clips as animations and event frames as photos with captions + buttons.
- **Tesla Integration (Optional)**:
  - **State of Charge (SoC) Display**: If a car is detected in a predefined location, the bot fetches the Tesla's SoC and displays it directly on the video highlight clip.
  - **Efficient Caching**: Caches the SoC in `tesla_soc.txt` and only queries the API periodically or when the cache is stale to avoid waking the vehicle unnecessarily.
- **Person Re-Identification (ReID)**:
  - **Who Crossed?** On gate crossings, the system optionally runs person re-identification using Intel's `person-reidentification-retail-0288` (OpenVINO) against a gallery in `person_of_interest/`.
  - **Line-Centered Sampling**: Samples person crops near the gate line tolerance every N frames and compares normalized embeddings via cosine similarity.
  - **Disk Cache**: Embeddings are cached to `temp/` so separate worker processes reuse precomputed vectors.
  - **Readable Output**: ReID score is appended to the Telegram message, and USERNAME is mentioned when a match exceeds the threshold. Optionally saves the best matching crops to the daily folder for manual review.
  - **Matched-Person Crop Targeting**: When multiple people cross the gate, crops belonging to the matched person are tagged with an `_m` filename suffix and reserved up to `REID_MATCHED_CROPS_MAX` slots in the saved top-K. Same-person crops are deduped via cosine similarity (`REID_SAME_PERSON_SIM`). AUTO_CONFIRM / AUTO_DECLINE flows and the manual confirm callbacks prefer matched-only crops, falling back to all crops when no `_m`-marked files are present.
  - **Per-Person Direction**: Each tracked entity's gate crossing direction (`up` / `down` / `both`) is recorded; the matched person's direction is used to drive the AUTO Reaction (Юху/Ех) even when other people cross in the opposite direction in the same video.
- **Optional Pose Estimation (POSE_ENABLED)**:
  - Post-tracking stage that runs a YOLO pose model on per-person crops collected during the event and writes one annotated clip per crossing person to `TEMP_DIR/YYYYMMDD/`.
  - Output filenames: `{video_stem}_pose_e{event}_p{display_id}_{direction}.mp4`. Clips are letterboxed to fixed even dimensions for libx264 compatibility.
  - Knobs: `POSE_MODEL_PATH`, `POSE_CONF_THRESHOLD`, `POSE_IMGSZ`, `POSE_CROP_PADDING`, `POSE_MAX_FRAMES_PER_PERSON`, `POSE_ABOVE_LINE_Y` (suppress crops whose bbox bottom falls below this Y, where pose quality degrades).
- **Performance & Stability**:
  - **Status-Routed Executor Lanes**: A single-worker pool handles CPU-bound motion detection, while the analysis stage is split across three independent pools routed by event status, so a slow lane never head-of-line blocks an instant one: `fast` (instant formatting — `no_motion`, `gate_crossing`), `llm` (local Ollama analysis — `no_person`, `no_significant_motion`), and `gemini` (Gemini API — `significant_motion`). Net effect: a gate crossing returns instantly while local-LLM clips are still processing, and Gemini runs in parallel with the local LLM rather than behind it. Pool sizes are set via `FAST_ANALYSIS_WORKERS` (default 2), `LOCAL_LLM_MAX_WORKERS` (default 1), and `GEMINI_MAX_WORKERS` (default 1).
  - **Graceful Shutdown & Auto-Restart**: Automatically restarts the script if any of the Python files is modified, with robust shutdown logic.
  - **Battery Monitoring**: Appends battery status to notifications if the device is on battery power (requires `psutil`).
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
  - `fastapi` (optional, for Log Dashboard)
  - `uvicorn` (optional, for Log Dashboard)
  - `jinja2` (optional, for Log Dashboard)

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

   # Optional: explicitly point to a dotenv file (otherwise ./.env is used)
   # DOTENV_PATH=/path/to/analyze-video/.env

   # Optional: enable motion profiler logs
   # ANALYZE_VIDEO_PROFILE=0

   # --- Object Detection (Optional) ---
   # Path to the exported OpenVINO model directory
   OBJECT_DETECTION_MODEL_PATH=models/best_openvino_model

   # --- Person Re-Identification (Optional) ---
   # Path to a folder with reference images of the person of interest
   REID_GALLERY_PATH=/path/to/person_of_interest

   # Optional: negative gallery (reject look-alikes)
   # REID_NEGATIVE_GALLERY_PATH=/path/to/person_of_interest_negative
   # REID_NEGATIVE_MARGIN=0.04

   # --- Tesla Integration (Optional) ---
   TESLA_EMAIL=your_tesla_account_email
   TESLA_REFRESH_TOKEN=your_tesla_api_refresh_token
   ```

5. **Configure Regions of Interest (ROI):**
    - Create resolution-specific ROI files in the `config/` directory:
       - `config/roi-4k.json`: ROI authored for 4K sources (baseline).
       - `config/roi-1080p.json`: ROI authored for 1080p sources (optional).
       - `config/roi.json`: legacy fallback if the resolution-specific file is not present.
    - You can use the script in `tools/gate_motion_detector.py` as a starting point to select an ROI for your video.
    - Each ROI file consolidates ROIs with clear purposes:
     - `motion_detection_roi`: polygon used for initial motion detection and for deriving the cropped analysis region baseline.
     - `tracker_roi` (optional): polygon defining the tracker’s working area; a padded bounding box around it is used to crop frames for faster tracking (falls back to `motion_detection_roi` if absent).
     - `person_tracker_roi` (optional): polygon filter for person detections; only persons whose bounding-box center lies inside this polygon are counted/tracked.
     - `line_y` (optional): line for gate crossing detection, if not specified value is taken from constants in `detect_motion.py`.

6. **(Optional) Configure AI Models and Prompts:**
    - Place your custom analysis instructions in `config/prompt.txt`.
    - Configure Gemini model selection via `config/gemini_models.env` (read on every analysis call, no restart needed):
       ```env
       MODEL_PRO=
       MODEL_MAIN=gemini-2.5-flash
       MODEL_FALLBACK=gemini-2.5-flash-lite
       MODEL_FINAL_FALLBACK=
       ```
      - If `MODEL_PRO` is set and the current hour is between 09 and 13, `MODEL_PRO` is used as the main model and `MODEL_MAIN` becomes the fallback.
      - If `MODEL_PRO` is empty or outside 09–13, `MODEL_MAIN` is used as main and `MODEL_FALLBACK` as fallback.
      - If `MODEL_FINAL_FALLBACK` is set, it will be used as a last resort.
      - Known codenames are displayed with responses: `gemini-3-flash-preview → 3FP`, `gemini-2.5-flash → 2.5F`, `gemini-2.5-flash-lite → 2.5FL`, `gemini-3.1-flash-lite-preview → 3.1FLP` (unknown models display `FF` for final fallback).

---

## Usage

1. **Start the application:**
   ```bash
   python main.py
   ```

2. **Place `.mp4` video files in the folder specified by the `VIDEO_FOLDER` environment variable.**
   The application will automatically detect, process, and send a notification to your Telegram chat.
   - Recommended layout under `VIDEO_FOLDER` is `YYYY/MM/DD/<video>.mp4`.

3. **Interact with the bot in Telegram.**
   - View highlight clips and insignificant motion snapshots.
   - Click the "Глянути" (View) or timestamp buttons to receive the full original video.

4. **(Optional) Run a one-off analysis for a single video:**
   ```bash
   python3 tools/run_detect_motion.py /path/to/video.mp4
   ```
   This prints a JSON summary (including up/down) and writes a highlight clip.

---

## How It Works

1. **File Monitoring:**
   - `watchdog` recursively monitors the `VIDEO_FOLDER`. When a new `.mp4` file is detected, it waits for the file size to stabilize before queuing it for analysis.

2. **Processing Pipeline:**
   - **Video Analysis (CPU-Bound Task):** The video is passed to the `motion_executor`.
     - **Motion Detection:** OpenCV analyzes frames within a cropped ROI to find initial motion, filtering out noise.
     - **Object Tracking:** If significant motion is found, the custom-trained YOLOv12 model tracks objects (people, cars) across frames.
       - **Event Classification:** The script determines the event type: `gate_crossing`, `significant_motion`, `no_significant_motion`, `no_person`, or `no_motion`.
     - **Artifact Generation:** A highlight clip (.mp4) and optional snapshots (.jpg) may be created in the `temp/` directory.
     - **Person ReID (Gate Crossings):**
       - During tracking, the analyzer samples person crops near the line tolerance band every `REID_SAMPLING_STRIDE` frames.
       - After a gate crossing is confirmed, these crops are compared against gallery embeddings using Intel's `person-reidentification-retail-0288` model.
       - Gallery embeddings are cached on disk at `temp/` per gallery path (configurable via `REID_CACHE_DIR`) to avoid recomputation across worker processes.
       - If a positive match is found, the gate message includes `XX%` and (optionally) the best crop is saved to the daily output folder.
   - **AI Analysis (I/O-Bound Task):** The result is routed to one of three status-based executor lanes (see "Status-Routed Executor Lanes" above).
     - Gate crossings and no-motion events return instantly (pure formatting, no external call).
     - `no_person` / `no_significant_motion` clips go to the local Ollama model for a description (at all hours; see "Local LLM Frame Analysis" below). If Ollama is unavailable, a placeholder is used.
     - `significant_motion` clips are sent to Gemini, with time-based model selection.

3. **Telegram Notification:**
   - A `telegram_lock` ensures that messages are sent or edited one at a time.
   - **Gate Crossing:** A special, high-priority message is sent immediately.
   - **Significant Motion:** A message is sent with the generated highlight clip and the AI description.
   - **Low Motion (`no_person` / `no_significant_motion`):** Sent as a single photo — the most detailed event frame — with the local-LLM description as the caption and a "Глянути" button (one self-contained, individually-actionable message). The frame is downscaled before upload (`TELEGRAM_FRAME_MAX_DIM`, default 1920px); the caption is clamped to Telegram's 1024-char limit. Disable with `SEND_INSIGNIFICANT_FRAMES=false`, which routes the description into the grouped text message instead.
   - **No Motion:** Events are grouped into a single, editable message to avoid spam.
   - **Leftover Frames (clip-bearing videos):** after a `gate_crossing` / `significant_motion` highlight is sent, any leftover low-motion event frame is analyzed by the local LLM in the background (the `llm` lane, so the highlight isn't delayed) and posted as a threaded reply — sent only when the LLM returns a description. Toggle with `SEND_GATE_EXTRA_FRAMES` (default on).
   - **Resilient Sending:** Animation delivery is retried non-blockingly at 5/10/15 minutes. If all retries fail (e.g., corrupted or oversized media), a final plain message with a “Глянути” button is sent so you still receive a notification. The frame-photo path likewise falls back to a plain text message with the button if the photo can't be sent. Highlight clips are preserved during retries and cleaned up after a successful send or after the final fallback.

4. **Callback Handling:**
   - When a button is clicked, the bot retrieves the corresponding full video file and sends it as a reply.

5. **Self-Monitoring & Auto-Restart:**
   - A separate `watchdog` instance monitors `*.py`. If any Python file is modified, it triggers a graceful shutdown and restarts the script.
   - Restart Recovery: To avoid losing files that appear during the restart window, the app writes a restart marker and, on startup, performs a short recovery pass:
       - It scans the `YYYY/MM/DD` day folder(s) that overlap the recovery window for new `.mp4` files and also consults a small processing ledger in `temp/processing_ledger.json`.
     - Files marked as `completed` in the ledger are skipped to prevent duplicates.
       - Files recorded as `queued`/`started`/`failed` are included in recovery as long as they still exist on disk.
     - The recovery window is controlled by `RESTART_RECOVERY_WINDOW_SECONDS` (default 180 seconds).
       - The ledger is pruned to the last 100 entries (most recent by `end_ts`/`start_ts`) to keep it small.
     - A concise summary is logged at the end of recovery with counts of candidates (by source), processed, and failures.

---

## Customization

- **Prompt:** Edit `config/prompt.txt` to change the analysis prompt sent to Gemini AI.
- **AI Models:** Edit `config/gemini_models.env` to control which Gemini models are used; changes apply without restarting the app.
- **Motion & Detection Parameters:** Adjust constants in `detect_motion.py` to fine-tune sensitivity. Key knobs:
  - `LINE_Y`: horizontal line Y position (gate; also used for car-below-line suppression)
  - `LINE_Y_TOLERANCE`: pixel tolerance around the line to suppress jitter near the threshold (default: 6 for 1080p, 12 for 4K)
  - `STABLE_MIN_FRAMES`: frames outside tolerance required to accept a side change (hysteresis; default: 2)
  - `DWELL_SECONDS`: minimum time on the new side to count mid-event flips; the last flip can be confirmed by final side (default: 2.0)
  - `TRACK_FULL_UNTIL_SECONDS`, `TRACK_SKIP_FROM_SECONDS`: speed/quality trade-offs for tracking/rendering long events (defaults: 6.0, 12.0)
  - `FAST_MOTION_STRIDE`, `FAST_TRACK_FULL_UNTIL_SECONDS`, `FAST_TRACK_SKIP_FROM_SECONDS`: fast-processing overrides for high-backlog mode (defaults: 3, 4.0, 8.0)
  - `MAX_EVENT_GAP_SECONDS`: max gap between motion frames before splitting into separate events (default: 3.0)
  - `MIN_EVENT_DURATION_SECONDS`, `MIN_INSIGNIFICANT_EVENT_DURATION_SECONDS`: event duration filters (defaults: 0.8, 0.2)
  - `PERSON_MIN_FRAMES`: minimum person-in-ROI frames required to keep an event (default: 10)
  - `MIN_CONTOUR_AREA`: motion contour area threshold for initial motion detection (default: 1800 for 1080p, 7200 for 4K)
  - `CONF_THRESHOLD`: detection confidence threshold for the YOLO model (default: 0.45)
  - `IOU_THRESHOLD`: IoU threshold passed to `model.track()` (default: 0.7)
  - `IMGSZ`: inference image size passed to `model.track()` / `model.predict()` (default: 640)
  - `TRACKER_CONFIG`: path to ByteTrack/BoTSORT config file (default: `config/tracker.yaml`)
  - `HIGHLIGHT_WINDOW_FRAMES`: frames to keep the red highlight after a crossing (default: 5)
  - `CROP_PADDING`: extra pixels around ROI for cropped analysis (default: 30 for 1080p, 60 for 4K)
  - `TRACK_ROI_ENABLED`: enable tracker ROI crop (default: True)
  - `TRACK_ROI_PADDING`: extra padding for tracker ROI crop (default: 10 for 1080p, 20 for 4K)
  - `SAVE_INSIGNIFICANT_FRAMES` (detect_motion.py): whether to save event frames to disk — needed for local LLM analysis, the Telegram photo, and the dashboard (default: True)
  - `STATIC_PERSON_MAX_MOVE_PX`: max allowed center drift (pixels) for a person to be considered static (default: 10)
  - `STATIC_PERSON_MIN_UPDATES`: minimum number of updates before a static person can be suppressed (default: 20)
  - `STATIC_PERSON_MAX_MEAN_CONF`: max mean confidence for a static person to be suppressed (default: 0.80)
  - `OBJECT_DETECTION_MODEL_PATH`: path to exported OpenVINO model (default: models/yolo12n_openvino_model)
  - Car speed (approximate):
    - `CAR_SPEED_BASE_MPP`: base meters-per-pixel scale (global calibration knob)
    - `CAR_SPEED_PERSPECTIVE_STRENGTH`, `CAR_SPEED_WIDEANGLE_EDGE_STRENGTH`: perspective/lens edge compensation
    - `CAR_SPEED_MEDIAN_WINDOW`, `CAR_SPEED_EMA_ALPHA`, `CAR_SPEED_DY_WEIGHT`: smoothing and jitter control
    - `CAR_SPEED_LABEL_MIN_KMH`, `CAR_SPEED_MAX_VALID_KMH`: label visibility and safety cap
    - `CAR_TRAIL_MAX_POINTS`, `CAR_TRAIL_THICKNESS`: trail length/appearance
    - Practical calibration: measure real distance on the road and count frames (`v_kmh = distance_m / (frames / fps) * 3.6`), then scale `CAR_SPEED_BASE_MPP` proportionally.
  - Car SpeedTrap:
    - `CAR_SPEEDTRAP_ENABLED`: enable/disable speedtrap overlay logic
    - `CAR_SPEEDTRAP_X1`, `CAR_SPEEDTRAP_X2`: vertical trap line positions in pixels
    - `CAR_SPEEDTRAP_DISTANCE_M`: real distance between `X1` and `X2` in meters
    - `CAR_SPEEDTRAP_MAX_VALID_KMH`: sanity cap for trap speed
    - `CAR_SPEEDTRAP_OVERLAY_EXTRA_GAP`: extra spacing between SpeedTrap and event overlay lines
  - `COLOR_PERSON`, `COLOR_CAR`, `COLOR_DEFAULT`, `COLOR_HIGHLIGHT`, `COLOR_LINE`: overlay colors (BGR tuples; defaults in code)
  - `OVERLAY_FONT_SCALE`, `OVERLAY_TEXT_THICKNESS`, `OVERLAY_BOX_THICKNESS`, `OVERLAY_LINE_THICKNESS`, `OVERLAY_LABEL_BG_HEIGHT`, `OVERLAY_PAD_X`, `OVERLAY_PAD_Y`: overlay appearance (resolution-dependent defaults)
  - Gate-crossing crop magnifier (lower-right live PiP of the person at the gate):
    - `GATE_CROP_OVERLAY_ENABLED`: enable/disable the magnifier (default: `true`)
    - `GATE_CROP_OVERLAY_WIDTH_FRAC`: PiP width as a fraction of the frame width (default: `0.18`)
    - `GATE_CROP_OVERLAY_BAND`: half-height (px) of the gate band within which the PiP is shown; `-1` derives it at runtime (default: `-1`)
    - `GATE_CROP_OVERLAY_BAND_SCALE`: multiplier applied to the derived band `(LINE_Y_TOLERANCE + REID_LINE_EXTRA_TOLERANCE)` — larger keeps the PiP visible longer (default: `4.0`)
    - `GATE_CROP_OVERLAY_UPPER_BODY`: show only the top 50% (head/torso) of the crop (default: `false`)
    - `GATE_CROP_OVERLAY_ALLOW_UPSCALE`: allow enlarging the PiP beyond the native crop size. If `false` — the magnifier crops from the native (4K) frame and only downscales, so it stays sharp (~2× vs the 1080p output); set `true` (and raise `WIDTH_FRAC`) for a bigger but upscaled PiP
  - `VIDEO_WRITER_PRESET`: libx264 encoder preset for highlight and pose clips (default: `faster`)
  - `TESLA_EMAIL`, `TESLA_REFRESH_TOKEN`: Tesla API credentials (master only; `detect_motion.py` only reads the cache)
  - `TESLA_SOC_FILE`: path to Tesla SoC cache file (default: `temp/tesla_soc.txt`)
  - `TESLA_SOC_DISPLAY_ENABLED`: show SoC overlay on highlight clips (default: `true`)
- **ReID Parameters (detect_motion.py):**
  - `REID_ENABLED`: enable/disable person re-identification (default: True)
  - `REID_MODEL_PATH`: path to Intel ReID model (default: models/reid/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml)
  - `REID_GALLERY_PATH`: folder with reference images (default: person_of_interest/)
  - `REID_THRESHOLD`: cosine similarity threshold for a positive match (default: 0.6)
  - `REID_SAMPLING_STRIDE`: sample every N frames near the line tolerance (default: 2)
  - `REID_LINE_EXTRA_TOLERANCE`: extra pixels beyond `LINE_Y_TOLERANCE` for ReID sampling (default: 20)
  - `REID_CROP_PADDING`: extra pixels around detected person boxes when creating ReID crops (default: 12)
  - `REID_MAX_SAMPLES`: cap on crops per video to keep inference snappy (default: 128)
  - `SAVE_REID_BEST_CROP`: save the best matching crop when matched (default: True)
  - `REID_TOP_K`: number of best, diverse crops to save per event (default: 3)
  - `REID_MATCHED_CROPS_MAX`: when ReID matches, reserve up to this many slots for diverse poses of the matched person (default: 2)
  - `REID_DIVERSITY_MIN_DIST`: min cosine distance between selected embeddings (default: 0.2)
  - `REID_SAME_PERSON_SIM`: cosine similarity above which two crops are treated as the same physical person for dedup across fragmented tracker IDs (default: 0.75)
  - `REID_NEGATIVE_GALLERY_PATH`: folder with negative reference images (default: person_of_interest_negative/)
  - `REID_NEGATIVE_MARGIN`: match must exceed negatives by at least this cosine margin (default: 0.08)
  - `REID_CACHE_DIR`: directory for precomputed embedding `.npz` cache files (default: `temp/`); point to a shared NAS path to reuse the master's cache on the worker
- **Pose Estimation Parameters (detect_motion.py):**
  - `POSE_ENABLED`: enable optional post-tracking pose stage (default: `false`)
  - `POSE_MODEL_PATH`: YOLO pose model path (default: `models/yolo11s-pose.engine`)
  - `POSE_CONF_THRESHOLD`: pose detection confidence (default: 0.25)
  - `POSE_IMGSZ`: pose inference image size (default: 640)
  - `POSE_CROP_PADDING`: extra pixels around person bbox for pose crops (default: 10)
  - `POSE_MAX_FRAMES_PER_PERSON`: cap on pose frames stored per tracked person (default: 400)
  - `POSE_ABOVE_LINE_Y`: skip pose crops whose bbox bottom is below this Y (avoids partial-body crops; default: 2100)
- **ReID Model Location:** By default, the Intel model XML is expected at `models/reid/intel/person-reidentification-retail-0288/FP16/person-reidentification-retail-0288.xml`. Adjust if your model path differs.
- **ROI:** Modify `config/roi-4k.json` / `config/roi-1080p.json` (or legacy `config/roi.json`) to change the monitored area.
- **Object Detection Model:** Choose pre-trained model between `models/*_openvino_model` or place your own exported YOLOv12 OpenVINO model there and adjust `OBJECT_DETECTION_MODEL_PATH`.
  - Follow instructions in [tools/finetuning/](tools/finetuning/) to train your own model if needed.

---

## Remote Worker (Optional)

Motion detection can be offloaded to a separate machine over HTTP, freeing the master's CPU/battery. The master falls back to local processing automatically if the worker is unavailable or has low battery.

Enable in the master's `.env`:

```env
WORKER_ENABLED=true
WORKER_URL=http://10.0.0.2:8741
WORKER_TIMEOUT=120
WORKER_MIN_BATTERY=5
```

The worker machine runs `uvicorn worker.server:app` from the same `analyze-video` directory, with its own `.env` pointing to the shared NAS mount. It can use a different model, confidence threshold, or tracker config than the master.

The worker `/health` endpoint reports: status, active/max tasks, battery percent, load averages (1m/5m/15m), memory usage, and CPU temperature (Package id 0 from coretemp).

### Wake-on-LAN

The master can automatically wake the worker when power is restored. When the worker health check fails and the master is plugged in, a WOL magic packet is sent (with a 5-minute cooldown between attempts).

Master `.env`:
```env
WORKER_WAKE_ON_LAN=true
WORKER_WAKE_ON_LAN_MAC=XX:XX:XX:XX:XX:XX
WORKER_WAKE_ON_LAN_IFACE_IP=YYY.YYY.YYY.YYY
```

The packet is sent via the `10.0.0.1` interface (configurable in `worker/client.py`).

Worker setup (Ubuntu Server):
```bash
# 1. Install ethtool
sudo apt install ethtool

# 2. Check/enable WOL on the ethernet interface
sudo ethtool enp3s0 | grep Wake-on
sudo ethtool -s enp3s0 wol g

# 3. Make persistent (survives reboot)
sudo tee /etc/systemd/network/50-wol.link << 'EOF'
[Match]
MACAddress=XX:XX:XX:XX:XX:XX

[Link]
WakeOnLan=magic
EOF

# 4. Enable "Wake on LAN" in BIOS/UEFI
# 5. Get MAC address: ip link show enp3s0 | grep ether
```

See [worker/README.md](worker/README.md) for full setup instructions.

---

## Local LLM Frame Analysis (Optional)

`no_person` and `no_significant_motion` clips (wind, shadows, sun) are described by a local [Ollama](https://ollama.com) vision model instead of Gemini, reserving the limited Gemini quota for `significant_motion`. The model analyzes the event frames `detect_motion` saves to the daily directory. If Ollama is unreachable, the pipeline falls back to the original placeholder messages.

### Worker setup

Run Ollama on the worker (the machine with the GPU) and pull a vision model:

```bash
# Listen on the LAN (not just localhost) and keep the model resident
OLLAMA_HOST=0.0.0.0:11434 ollama serve
ollama pull qwen3-vl:2b-instruct
```

The client sends `keep_alive` per request (from `OLLAMA_KEEP_ALIVE`), so the model stays loaded between calls.

### Enable in the master's `.env`

```env
LOCAL_FRAME_ANALYSIS_ENABLED=true          # default true
OLLAMA_URL=http://10.0.0.2:11434           # worker's Ollama endpoint
OLLAMA_MODEL=qwen3-vl:2b-instruct          # vision model
OLLAMA_TIMEOUT=60
OLLAMA_KEEP_ALIVE=-1                        # keep the model resident (or e.g. 30m)
```

Prompts are read on every call (edits apply without a restart) and default to English: `config/prompt_frame.txt` (single frame) and `config/prompt_frame_multi.txt` (multi-frame). Override the paths with `OLLAMA_PROMPT_FILE` / `OLLAMA_MULTI_PROMPT_FILE` to swap prompts without editing the defaults.

### Two operating modes

**1. Single model.** One vision model produces the final description. With `OLLAMA_MULTI_FRAME=true` (default), a clip's frames go to the model in one multi-image call (sampled down to `OLLAMA_MULTI_MAX_FRAMES`, default 4); set it to `false` for legacy per-frame analysis plus a text-only merge. Use a vision model whose target-language output you trust, and point the prompt-file env vars at a prompt in that language.

Example — a larger model (`gemma4:12b`) writing a longer, narrative Ukrainian description directly (requires a recent Ollama with gemma4 support):

```env
OLLAMA_MODEL=gemma4:12b-it-q4_K_M
OLLAMA_REFINE_MODEL=                        # empty — single-model mode
OLLAMA_PROMPT_FILE=config/prompt_frame_long_uk.txt
OLLAMA_MULTI_PROMPT_FILE=config/prompt_frame_long_uk.txt
OLLAMA_NUM_PREDICT=-1                        # let the model finish its narrative
OLLAMA_MAX_CHARS=4000                        # allow long-form output (default 300 would retry it as a ramble)
OLLAMA_THINK=false                           # reasoning on is far slower; off is acceptable for these low-priority clips
OLLAMA_TEMPERATURE=0.8
OLLAMA_REPEAT_PENALTY=1.1                     # lower than default — kinder to long prose
OLLAMA_TIMEOUT=360                           # long-form generation is slow on CPU-bound hardware
OLLAMA_KEEP_ALIVE=30m
```

**2. Two-stage (recommended for clean Ukrainian).** Set `OLLAMA_REFINE_MODEL` to enable a two-stage pipeline: a fast vision model describes the frames in English (grounding only), then a dedicated language model rewrites them into one cheeky, idiomatic, russism-free Ukrainian sentence via a text-only call. This gives the cleanest Ukrainian, because a native-language text model writes the output rather than the vision model.

```env
# Stage 1 — vision (English grounding)
OLLAMA_MODEL=qwen3-vl:2b-instruct
OLLAMA_PROMPT_FILE=config/prompt_frame.txt

# Stage 2 — refine (Ukrainian)
OLLAMA_REFINE_MODEL=hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M
OLLAMA_REFINE_NUM_GPU=0                     # run the refiner on CPU, leaving the GPU to the vision model
OLLAMA_REFINE_KEEP_ALIVE=-1                 # keep the (0-VRAM) refiner resident; avoids its cold load
OLLAMA_MAX_CHARS=450                        # Ukrainian runs longer than the default 300
OLLAMA_TIMEOUT=200
```

The refine prompt is `config/prompt_frame_refine_uk.txt` (uses a `{descriptions}` placeholder); override with `OLLAMA_REFINE_PROMPT_FILE`.

### Tuning knobs

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_ATTEMPTS` | `2` | Total tries per call (retries on request error, empty, or over-length response) |
| `OLLAMA_IMAGE_MAX_DIM` | `1280` | Downscale a frame's longest side before sending (a 4K frame's vision-patch count is slow to encode); `0` disables |
| `OLLAMA_TEMPERATURE` / `OLLAMA_TOP_K` / `OLLAMA_TOP_P` | `0.8` / `20` / `0.92` | Sampling; the tight top_k/top_p clamp keeps the high temperature coherent |
| `OLLAMA_NUM_PREDICT` | `120` | Output length cap (`-1` for unbounded — required for reasoning models) |
| `OLLAMA_REPEAT_PENALTY` | `1.3` | Breaks repetition loops |
| `OLLAMA_MAX_CHARS` | `300` | A longer response is treated as a rambling loop and retried; raise it for longer-form output |
| `OLLAMA_THINK` | unset | Set `true` for reasoning models (use with `OLLAMA_NUM_PREDICT=-1`); better grounding at a large latency cost |
| `OLLAMA_HOKKU_PROBABILITY` | `0` | Probability (0–1) a `{format}`-placeholder prompt is answered as a hokku instead of prose (see "Random output format" below) |

### Telegram delivery

The description is delivered as a single photo (the most detailed event frame) with the description as caption and a "Глянути" full-video button. The frame is downscaled before upload (Telegram re-compresses photos anyway), and the caption is clamped to Telegram's 1024-char hard limit — keep the prompt's output under ~900 chars to leave room for the appended object counts / battery suffix.

| Variable | Default | Purpose |
|---|---|---|
| `SEND_INSIGNIFICANT_FRAMES` | `true` | Send the event-frame photo for `no_person`/`no_significant_motion`. `false` → the description is delivered as a grouped text message instead (no image) |
| `SEND_GATE_EXTRA_FRAMES` | `true` | For `gate_crossing`/`significant_motion` videos, post any leftover low-motion frame (LLM-analyzed) as a threaded reply to the highlight, after it's sent |
| `TELEGRAM_FRAME_MAX_DIM` | `1920` | Downscale the frame's longest side before upload (`0` disables); `1280` roughly halves the size again |
| `TELEGRAM_FRAME_JPEG_QUALITY` | `88` | JPEG quality for the downscaled frame |

The on-disk frame is left full-resolution for the dashboard; only the uploaded copy is downscaled.

### Prompt files

All prompts live in `config/` and are re-read on every call (edits apply without a restart). The frame prompts below can each be overridden with the env var shown; the Gemini prompt path is fixed.

| File | Lang | Used by | Override env var |
|---|---|---|---|
| `prompt.txt` | UK | **Gemini** analysis of `significant_motion` highlight clips | *(fixed path)* |
| `prompt_frame.txt` | EN | Single-frame local analysis; also **stage 1 (vision)** of the two-stage pipeline | `OLLAMA_PROMPT_FILE` |
| `prompt_frame_multi.txt` | EN | Multi-frame local analysis (one multi-image call) | `OLLAMA_MULTI_PROMPT_FILE` |
| `prompt_frame_combine.txt` | EN | Legacy per-frame **merge** (only when `OLLAMA_MULTI_FRAME=false`) | `OLLAMA_COMBINE_PROMPT_FILE` |
| `prompt_frame_uk.txt` | UK | Ukrainian single-frame variant (single-model UK output) | `OLLAMA_PROMPT_FILE` |
| `prompt_frame_multi_uk.txt` | UK | Ukrainian multi-frame variant | `OLLAMA_MULTI_PROMPT_FILE` |
| `prompt_frame_combine_uk.txt` | UK | Ukrainian legacy merge variant | `OLLAMA_COMBINE_PROMPT_FILE` |
| `prompt_frame_refine_uk.txt` | UK | **Stage 2 (refine)** of the two-stage pipeline; has a `{descriptions}` placeholder | `OLLAMA_REFINE_PROMPT_FILE` |
| `prompt_frame_long_uk.txt` | UK | Long-form narrative Ukrainian for a single larger model (e.g. gemma4:12b); contains a `{format}` placeholder | `OLLAMA_PROMPT_FILE` / `OLLAMA_MULTI_PROMPT_FILE` |
| `prompt_frame_fmt_prose_uk.txt` | UK | Prose format snippet substituted into `{format}` (the default branch) | `OLLAMA_FORMAT_PROSE_FILE` |
| `prompt_frame_fmt_hokku_uk.txt` | UK | Hokku format snippet substituted into `{format}` (the random branch) | `OLLAMA_FORMAT_HOKKU_FILE` |

The single-frame vs multi-frame prompt is chosen automatically per clip (1 frame → single, 2+ → multi). The `_uk` and `long_uk` files are alternatives you point the same env vars at — only the prompts referenced by your active config are used.

#### Random output format (prose vs hokku)

A prompt containing a `{format}` placeholder (like `prompt_frame_long_uk.txt`) gets that placeholder replaced **per call** with either the prose or the hokku snippet, chosen by a real coin flip in code — `OLLAMA_HOKKU_PROBABILITY` (0–1, default `0` = always prose). The randomness must live in code: an LLM can't make a fair random choice (greedy decoding collapses "pick a number 1–20" to a constant, almost always 13), so asking the model to roll a die never varies. The shared scene/subject/anti-russism rules stay in the body; only the format line is swapped. Set `OLLAMA_HOKKU_PROBABILITY=1` to force hokku for testing. No-op for prompts without `{format}`.

---

## Log Dashboard

A lightweight web dashboard that reads existing log files and provides per-day insights, a readable log viewer, and a JSON API for the [Android dashboard app](https://github.com/alivespirit/analyze-video-dashboard).

- HTML routes:
   - `/` lists all available days (today first)
   - `/today` shows today (or latest available) with filters
   - `/day/{YYYY-MM-DD}` shows a specific day with filters
   - `/stats` shows aggregated stats across all log days
- JSON API routes (for Android app):
   - `/api/days` — list of available log days
   - `/api/today/videos?day=` — per-video summary with status, ReID, frames indicator, speed, pipeline-error flag
   - `/api/today/video/{basename}/logs?day=` — log entries per video
   - `/api/today/video/{basename}/reid-crops` — ReID crop image URLs
   - `/api/today/video/{basename}/frames` — insignificant/no_person frame URLs
   - `/api/today/video/{basename}/highlight` — highlight clip URL
   - `/api/today/video/{basename}/pose` — pose clip URLs (when POSE_ENABLED)
   - `/api/today/video/{basename}/full` — full source video URL (only when the file currently exists)
   - `/api/today/gate-crossings?day=` — videos with ReID crops: basename, time, direction, persons up/down, away/back, scores, crop URLs
   - `/api/today/stats?day=` — aggregated stats for a day; today's open away intervals carry elapsed `dur` and an `ongoing` flag
   - `/api/stats/overall` — overall stats with heatmaps; weekday and time-of-day heatmaps include cached history beyond log retention; weekday cells include per-bin occurrence lists
   - `/api/stats/reid` — ReID accuracy metrics per day (TP/FP/FN, precision/recall/F1, match score) + 7-day MA + per-day events with crop URLs; persisted in `temp/reid_metrics_cache.json` and not pruned
   - `/api/monitoring` — system monitoring (CPU, RAM, battery, worker health)
   - `/api/events/latest?since=` — away/back events for notifications, with optional `next_prediction` (same-weekday pattern, ≥30% confidence, within 3 h horizon)
   - `/api/reid/copy` (POST) — copy ReID crop to gallery
   - `/api/gallery/{positive|negative}/{filename}` — GET serves gallery reference crops; DELETE removes them (cache rebuilds automatically on next ReID run)
   - `/api/image/{basename}`, `/api/highlight/{basename}` — serve images/clips
- Per-day views: timestamp, severity, video basename, message
- Filters:
   - Severity (single-select)
   - Status (multi-select, CSV in the URL like `status=no_motion,gate_crossing`)
   - Gate direction (up/down). Videos detected as `both` match both filters.
   - Video (via `video=<basename>` or by clicking a video in the logs)
- Status Counts with totals and filtered vs total video count
- Gate Crossings tile with up/down counts and a compact Away/Back interval list (when present)
- Collapsible Per-Video Summary: start time, status, raw events, processing time + optional chips (gate direction, away/back reaction state, ReID results when present)
- Processing Times Chart: evenly spaced bars, colored by status, with min/avg/max guide lines; bars are clickable to jump to logs; hour boundaries are marked
- Responsive UX: stacked log entries on mobile and floating up/down buttons on mobile

### Stats page (`/stats`)

Aggregated view across all discovered log files (current + rotated):

- Events heatmaps for Away/Back by time-of-day (06:00–24:00) and by weekday
- Total Unique Videos per day (stacked by status, including `unknown` when needed)
- Average Motion Detection time per day
- Average Full Processing time per day

### Video Playback

- Embedded player: Clicking a video name in Per-Video Summary opens an embedded HTML5 player inline beneath the selected row. No full page reload.
- Timestamp links: Clicking the start time jumps to the first log entry for that video.
- Preserves filters: Opening a video keeps current filters (severity/status/gate) unchanged. The URL is updated with `play=<basename>` via the History API for sharability and back/forward navigation.
- Gate indicator: An arrow (↑/↓/↕) appears in the Status column indicating gate direction (both counts as up and down).
- Requirements:
   - Set `VIDEO_FOLDER` in your environment to the root folder where original `.mp4` files reside (monitored recursively).
   - The dashboard serves files via `GET /video/{basename}`. Only `.mp4` basenames are allowed; the first matching file is returned.
- Route details:
   - `play=<basename>` query parameter triggers the embedded player (e.g., `/today?play=YYYYMMDD_HHMMSS.mp4`).
   - The Per-Video Summary remains open when a player is shown and highlights the currently opened video row. Scrolling positions the player so one table row is visible above it (accounting for the sticky header).

### Chart Hour Boundaries

- Hour separators are drawn between bars when the next video’s hour differs from the current one.
- Separators use dashed styling and include small hour labels (e.g., `2h`) for clarity.

### Collapsible State Persistence

- The open/closed state of collapsible sections (Per-Video Summary, Logs) is remembered via localStorage.
- Navigating via the header (Home/All Days) clears the persisted state to start fresh.

### Scrolling Behavior

- Player focus uses native smooth scrolling and aligns the player with one table row visible above it.

### Filtering

- Multi-select Status: Click status badges or legend items to add/remove multiple statuses. The URL uses a single `status` CSV (e.g., `status=no_motion,gate_crossing`).
- Full counts when filtered: Status Counts always show totals for the day, even when filters are applied.
- Gate filter: `up`/`down` supported; `both` is included in counts for both directions when detected.
- Clear control: When any filter is active, a small Clear chip appears next to the "Filtered videos" summary and resets all filters.

### Enable from main.py

Set in `.env`:

```bash
ENABLE_LOG_DASHBOARD=true
LOG_DASHBOARD_PORT=8192        # optional, defaults to 8192
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
