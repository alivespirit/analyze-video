# Analyze-Video Log Dashboard

A lightweight FastAPI server that parses your application logs and serves a simple dashboard for daily insights.

## Features

- Per-day log viewer with: timestamp, severity, video basename, message
- Filters: Severity and Status (e.g., `no_motion`, `gate_crossing`, `no_significant_motion`, `error`)
- Status Counts tile shows totals and (when filtered) filtered vs total for the day
- Per-Video Summary (collapsible):
	- Start time: first "New file detected" timestamp (HH:MM:SS)
	- Status, raw events, processing time (prefers Motion Detection time, falls back to Full Processing time)
- Processing Times Chart:
	- Bars evenly spaced across full width (one per video)
	- Bar colors by status with a compact legend
	- Click a bar to jump to the first corresponding log entry
- Readable log list with stable, per-video colors for quick visual grouping
- “Available Days” page lists today first, then the rest in descending order

### Video Playback

- Click a video name in the Per-Video Summary to open an embedded player.
- Click the timestamp to jump to the first log entry of that video.
- Gate direction chip: Up/Down/Both arrows (↑/↓/↕) render next to the status badge.
- Filters are preserved when opening a video; the current severity/status/gate selections remain active.
- The summary stays expanded when the player is visible, and the current video row is highlighted.

Requirements:
- Set `VIDEO_FOLDER` to the root directory containing your original `.mp4` files. The server searches it recursively.
- Files are served via `GET /video/{basename}`; only `.mp4` basenames are allowed.
- To open the player directly, use `play=<basename>` in the query string (e.g., `/today?play=clip123.mp4#player`).

## Run

1) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2) (Optional) Configure log directory and basename:

```bash
export LOG_PATH=/path/to/logs        # directory containing video_processor.log and rotations
export LOG_BASENAME=video_processor.log  # optional; defaults to video_processor.log
export VIDEO_FOLDER=/path/to/videos  # required for embedded video playback (.mp4)
```

3) Start the server (standalone) or enable auto-start from main:

```bash
python -m uvicorn tools.log_dashboard.app:app --port 8000 --reload
```

Open http://127.0.0.1:8000 and select a day.

### Auto-start with main.py

Set in your `.env`:

```bash
ENABLE_LOG_DASHBOARD=true
LOG_DASHBOARD_PORT=8000        # optional, defaults to 8000
LOG_DASHBOARD_HOST=0.0.0.0     # optional, defaults to 0.0.0.0
```

When `ENABLE_LOG_DASHBOARD` is true, `main.py` starts the dashboard in a background thread and stops it on graceful shutdown. By default, it listens on `0.0.0.0` (LAN-visible).

## Log discovery

- Current day: `video_processor.log` (filtered by selected date)
- Rotated days: `video_processor_YYYY-MM-DD.log`

## Status colors

- `no_motion`: slate
- `no_significant_motion`: amber
- `significant_motion`: violet
- `gate_crossing`: green
- `error`: red

## Notes

- Metrics are computed from existing messages; extending the parser is straightforward in [tools/log_dashboard/app.py](tools/log_dashboard/app.py).
- Environment variables: `LOG_PATH` (directory) and optional `LOG_BASENAME` (default `video_processor.log`).
- The Per-Video Summary table is collapsible (open by default).
- Chart bars link to the first log entry of the related video and highlight the target entry.