# Worker

Remote motion detection worker for Споглядайко.

Runs on a separate machine (e.g. a laptop with a GPU) and exposes an HTTP API that the master calls to offload `detect_motion`. The master falls back to local processing if the worker is unavailable.

---

## How It Works

1. Master sends a POST `/detect-motion` with the master-perspective paths to the video and output directory.
2. Worker translates the paths from master format (`C:\NAS\...`) to its own CIFS mount (`/mnt/nas/...`).
3. Worker copies the video to a local temp directory (avoids slow CIFS I/O during processing).
4. `detect_motion` runs locally on the worker hardware.
5. Output files are copied back to the CIFS mount.
6. Result paths are translated back to master-perspective and returned as JSON, along with captured log lines.
7. Master replays the log lines into its own log file, inserting `[W]` after the `[filename]` bracket.

The worker runs up to `WORKER_MAX_CONCURRENT` (default: 2) videos in parallel using a thread pool. The master dispatches asynchronously (no blocking executor), so it can saturate all worker slots.

---

## Setup

### 1. Clone the repository on the worker machine

The worker needs the same `analyze-video` directory since it imports `detect_motion` directly:

```bash
git clone https://github.com/alivespirit/analyze-video.git
cd analyze-video
python3 -m venv .venv && source .venv/bin/activate
pip install -r worker/requirements.txt
```

### 2. Mount the NAS

The worker must have the shared NAS accessible, e.g. via CIFS:

```bash
# /etc/fstab or systemd automount — mount at /mnt/nas
```

### 3. Create a `.env` in the worker's `analyze-video` directory

```env
MASTER_PATH_PREFIX=C:\NAS\
WORKER_PATH_PREFIX=/mnt/nas/

# Worker settings
WORKER_PORT=8741
WORKER_MAX_CONCURRENT=2

# Detection config (different from master)
OBJECT_DETECTION_MODEL_PATH=models/yolo12s.engine
CONF_THRESHOLD=0.35
TRACKER_CONFIG=tracker-yolo12s.yaml
IOU_THRESHOLD=0.5
TRACK_ROI_ENABLED=false
IMGSZ=640

# ReID config
REID_GALLERY_PATH=/mnt/nas/analyze-video/person_of_interest
REID_NEGATIVE_GALLERY_PATH=/mnt/nas/analyze-video/person_of_interest_negative
REID_MODEL_PATH=models/reid/intel/person-reidentification-retail-0286/FP16/person-reidentification-retail-0286.xml
REID_THRESHOLD=0.5

# Read Tesla SoC cache from NAS (written by master)
TESLA_SOC_FILE=/mnt/nas/analyze-video/temp/tesla_soc.txt
```

> **Tip:** Point `REID_GALLERY_PATH`, `REID_NEGATIVE_GALLERY_PATH`, and `TESLA_SOC_FILE` to the NAS paths so the worker always uses the same data as the master without duplication. For the ReID embedding cache, either set `REID_CACHE_DIR` to the master's `temp/` on CIFS to reuse prebuilt NPZ files, or leave it unset so the worker builds its own local cache (takes ~35 s on first run with a large gallery).

### 4. Install the systemd service

```bash
sudo cp worker/worker.service /etc/systemd/system/analyze-video-worker.service
# Edit User= and WorkingDirectory= if they differ from the defaults in the file
sudo systemctl daemon-reload
sudo systemctl enable --now analyze-video-worker.service
```

### 5. Configure the master

Add to the master's `.env`:

```env
WORKER_ENABLED=true
WORKER_URL=http://10.0.0.2:8741
WORKER_TIMEOUT=120
WORKER_HEALTH_CACHE_SECONDS=30
WORKER_MIN_BATTERY=5       # skip worker if its battery is below this %

# Wake-on-LAN (optional) — wake worker when it's offline and master is plugged in
WORKER_WAKE_ON_LAN=true
WORKER_WAKE_ON_LAN_MAC=XX:XX:XX:XX:XX:XX
```

---

## Starting Manually

Run from the `analyze-video` directory:

```bash
uvicorn worker.server:app --host 0.0.0.0 --port 8741 --timeout-keep-alive 120
```

`--timeout-keep-alive 120` raises the idle connection timeout from uvicorn's default of 5 s, preventing TCP resets on long-running detection jobs.

---

## API

### `GET /health`

Returns worker status. Used by master for availability checks and by the Android dashboard for the Monitoring tab.

```json
{
  "status": "ok",
  "active_tasks": 1,
  "max_tasks": 2,
  "battery_percent": 87.0,
  "load_avg_1m": 1.2,
  "load_avg_5m": 0.8,
  "load_avg_15m": 0.5,
  "memory_percent": 45.0,
  "memory_used_mb": 3200,
  "memory_total_mb": 7892,
  "cpu_temp_c": 44.0
}
```

- Master skips the worker if `battery_percent` is below `WORKER_MIN_BATTERY`.
- `load_avg_*`: from `os.getloadavg()` (Linux only).
- `cpu_temp_c`: Package id 0 from `psutil.sensors_temperatures()` coretemp (null if unavailable).

### `POST /detect-motion`

Request body:

```json
{
  "video_path": "C:\\NAS\\videos\\2025\\01\\01\\video.mp4",
  "output_dir": "C:\\NAS\\videos\\2025\\01\\01\\output",
  "fast_processing": false
}
```

Response:

```json
{
  "result": { ... },
  "logs": ["2025-01-01 12:00:00 - INFO - [video.mp4] ..."]
}
```

`result` has the same shape as `detect_motion()` return value, with all paths translated back to master-perspective (Windows format).

---

## Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `MASTER_PATH_PREFIX` | `C:\NAS\` | NAS path prefix on the master (Windows) |
| `WORKER_PATH_PREFIX` | `/mnt/nas/` | NAS mount point on the worker |
| `WORKER_MAX_CONCURRENT` | `2` | Max simultaneous videos processed |
| `REID_CACHE_DIR` | `temp/` (relative to script) | Directory for ReID embedding cache; point to master's `temp/` on CIFS to reuse prebuilt cache |
| `WORKER_WAKE_ON_LAN` | `false` | Send WOL magic packet when worker health check fails and master is plugged in |
| `WORKER_WAKE_ON_LAN_MAC` | (empty) | Worker's ethernet MAC address for WOL |

All `detect_motion` env vars work on the worker the same way as on the master (`CONF_THRESHOLD`, `IOU_THRESHOLD`, `IMGSZ`, `TRACKER_CONFIG`, `OBJECT_DETECTION_MODEL_PATH`, `REID_*`, etc).

WOL packets are sent via the `10.0.0.1` interface with a 5-minute cooldown. The worker must have Wake-on-LAN enabled in BIOS and via `ethtool -s <iface> wol g`.

---

## Files

| File | Description |
|---|---|
| `server.py` | FastAPI server — path translation, log capture, result copy |
| `client.py` | Master-side client — health check, async dispatch, log replay, local fallback |
| `worker.service` | systemd service unit |
| `requirements.txt` | Worker-only Python dependencies |
