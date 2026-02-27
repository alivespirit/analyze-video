# Analyze-Video Log Dashboard

A lightweight FastAPI server that parses your application logs and serves a simple dashboard for daily insights.

## Features

- Per-day log viewer (`/today`, `/day/{YYYY-MM-DD}`) with: timestamp, severity, video basename, message
- Filters:
	- Severity (single-select)
	- Status (multi-select, comma-separated in the URL like `status=no_motion,gate_crossing`)
	- Gate direction (up/down). Videos detected as `both` match both filters.
	- Video (click a video name in logs to filter by it, or use `video=<basename>`)
- Status Counts tile shows totals for the whole day and, when filtered, shows filtered vs total video count
- Gate Crossings tile shows up/down counts and a compact Away/Back interval list (when present)
- Per-Video Summary (collapsible):
	- Start time: first "New file detected" timestamp (HH:MM:SS)
	- Status, raw events, processing time (prefers Motion Detection time, falls back to Full Processing time)
	- Optional chips for gate direction, away/back reaction state, and ReID results (when present in logs)
- Processing Times Chart:
	- Bars evenly spaced across full width (one per video)
	- Bar colors by status with a compact legend
	- Click a bar to jump to the first corresponding log entry
	- Hour boundary separators + labels when the next video starts in a new hour
- Readable log list with stable, per-video colors for quick visual grouping
- “Available Days” page lists today first, then the rest in descending order

### Stats page (`/stats`)

The Stats page aggregates across all discovered log files (current + rotated) and shows:

- **Events (Away/Back) by Time of Day**: a 06:00–24:00 heatmap.
	- 15-minute bins.
	- Intensity represents the **% of days** that had at least one Away/Back event in that time window.
	- Includes an hourly marginal strip summarizing how often events happen per hour.
- **Total Unique Videos** per day as stacked bars by final status (includes an `unknown` bucket when needed so totals match).
- **Average Motion Detection Time** per day.
- **Average Full Processing Time** per day.
- **Away by Weekday** and **Back by Weekday** heatmaps (hourly bins, normalized within each weekday as % of days).

### Video Playback

- Click a video name in the Per-Video Summary to open an embedded player inline (no page reload). The player appears directly beneath the selected row.
- Click the timestamp to jump to the first log entry of that video.
- Gate direction chip: Up/Down/Both arrows (↑/↓/↕) render next to the status badge.
- Filters are preserved when opening a video; the current severity/status/gate selections remain active. The URL is updated with `play=<basename>` via History API for sharability and back/forward navigation.
- The summary stays expanded when the player is visible, and the current video row is highlighted. Scrolling positions the player so one table row is visible above it (accounting for the sticky header).

Requirements:
- Set `VIDEO_FOLDER` to the root directory containing your original `.mp4` files. The server searches it recursively.
- Files are served via `GET /video/{basename}`; only `.mp4` basenames are allowed.
- To open the player directly, use `play=<basename>` in the query string (e.g., `/today?play=clip123.mp4`).

### Chart Hour Boundaries

- Hour separators are drawn between bars when the next video’s hour differs from the current one.
- Separators use dashed styling for clarity and include small hour labels (e.g., `2h`).

### Collapsible State Persistence

- The open/closed state of collapsible sections (like Per-Video Summary and Logs) is remembered via localStorage.
- Navigating via the header’s Home/All Days clears the persisted state so you start fresh.

### Scrolling Behavior

- Player focus uses native smooth scrolling and aligns the player with one table row visible above it.

### Filtering

- Status is multi-select: toggle badges or click legend items to add/remove statuses. The query uses a single CSV param like `status=no_motion,gate_crossing`.
- Counts remain full-day while filtering; the Status Counts tile always shows totals for all statuses.
- Gate filter supports `up` and `down`, with `both` counted in both directions when present.
- You can filter to a single video via `video=<basename>` (also available by clicking a video in the log list).
- The header shows "Filtered videos" when any filter is active; a Clear control resets all filters back to the full day view.

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
- `no_person`: yellow
- `significant_motion`: blue
- `gate_crossing`: green
- `error`: red

## Notes

- Metrics are computed from existing messages; extending the parser is straightforward in [tools/log_dashboard/app.py](tools/log_dashboard/app.py).
- Environment variables: `LOG_PATH` (directory) and optional `LOG_BASENAME` (default `video_processor.log`).
- The Per-Video Summary table is collapsible (open by default).
- Chart bars link to the first log entry of the related video and highlight the target entry.