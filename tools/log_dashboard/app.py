import os
import re
import colorsys
from datetime import datetime, date
from typing import List, Dict, Optional, Tuple

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape


# Config: align defaults with main.py
LOG_PATH = os.getenv("LOG_PATH", default="")
LOG_DIR = LOG_PATH if LOG_PATH else os.getcwd()
LOG_BASE = os.getenv("LOG_BASENAME", default="video_processor.log")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER", default=None)

# Patterns used by CustomTimedRotatingFileHandler in main.py
ROTATED_PATTERN = re.compile(r"^(?P<base>.+)_((?P<date>\d{4}-\d{2}-\d{2})).log$")

# Log line pattern: "YYYY-MM-DD HH:MM:SS - LEVEL - message"
LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (?P<level>[A-Z]+) - (?P<message>.*)$"
)

# Optional video basename in message: "[basename] Content..."
MSG_VIDEO_RE = re.compile(r"^\[(?P<video>[^\]]+)\]\s*(?P<content>.*)$")

# Metrics message patterns
STATUS_RE = re.compile(r"Motion detection complete\. Status: (?P<status>[a-z_]+)")
# Generic status line matcher (covers non-MD contexts like telegram failures)
STATUS_ANY_RE = re.compile(r"Status:\s*(?P<status>[a-z_]+)")
FULL_TIME_RE = re.compile(r"Full processing took (?P<seconds>[0-9]+\.[0-9]+|[0-9]+) seconds")
RAW_EVENTS_RE = re.compile(r"Found (?P<count>\d+) raw motion event\(s\)")
MD_TIME_RE = re.compile(r"Motion detection took (?P<seconds>[0-9]+\.[0-9]+|[0-9]+) seconds")
GATE_RE = re.compile(r"Gate crossing detected! Direction: (?P<dir>up|down|both)")
NEW_FILE_RE = re.compile(r"^New file detected:")
AWAY_RE = re.compile(r"Reaction detected: object went away")
BACK_RE = re.compile(r"Reaction detected: object came back")
REACTION_REMOVED_RE = re.compile(r"Reaction removed\.")
# ReID result patterns (old and new)
# Old:  "ReID result: matched=True/False, best_score=0.xxx, threshold=..."
# New:  "ReID result: matched=True/False, pos=0.xxx, neg=0.xxx, delta=0.xxx, thr=0.xxx, margin=0.xxx."
REID_RESULT_RE_OLD = re.compile(
    r"ReID result:\s*matched=(?P<matched>True|False),\s*best_score=(?P<score>[0-9]*\.?[0-9]+),\s*threshold=(?P<thresh>[0-9]*\.?[0-9]+)"
)
REID_RESULT_RE_NEW = re.compile(
    r"ReID result:\s*matched=(?P<matched>True|False),\s*pos=(?P<pos>[0-9]*\.?[0-9]+),\s*neg=(?P<neg>[0-9]*\.?[0-9]+),\s*delta=(?P<delta>-?[0-9]*\.?[0-9]+),\s*thr=(?P<thresh>[0-9]*\.?[0-9]+),\s*margin=(?P<margin>[0-9]*\.?[0-9]+)"
)


env = Environment(
    loader=FileSystemLoader(os.path.join(os.path.dirname(__file__), "templates")),
    autoescape=select_autoescape(["html", "xml"]),
)

app = FastAPI(title="Analyze-Video Log Dashboard")


def list_log_files() -> Dict[str, str]:
    """Return map of date (YYYY-MM-DD) to file path.

    Supports rotated files "video_processor_YYYY-MM-DD.log" and current file "video_processor.log".
    """
    date_to_path: Dict[str, str] = {}
    try:
        for fname in os.listdir(LOG_DIR):
            fpath = os.path.join(LOG_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            if fname == LOG_BASE:
                # Current log; determine date by first matching line
                day = detect_log_day(fpath)
                if day:
                    date_to_path[day] = fpath
                continue
            # Rotated pattern: video_processor_YYYY-MM-DD.log
            m = ROTATED_PATTERN.match(fname)
            if m:
                base = m.group("base")
                if not base.endswith(os.path.splitext(LOG_BASE)[0]):
                    continue
                day = m.group("date")
                date_to_path[day] = fpath
    except FileNotFoundError:
        pass
    return dict(sorted(date_to_path.items()))


def detect_log_day(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf8", errors="ignore") as f:
            for line in f:
                m = LINE_RE.match(line.rstrip("\n"))
                if m:
                    ts = m.group("ts")
                    return ts.split(" ")[0]
    except Exception:
        return None
    return None


def parse_log_lines(path: str, day: str) -> List[Dict]:
    """Parse log file and return list of entries for the specific day.

    Each entry: {ts: datetime, level: str, video: Optional[str], content: str}
    """
    entries: List[Dict] = []
    with open(path, "r", encoding="utf8", errors="ignore") as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = LINE_RE.match(line)
            if not m:
                continue
            ts_str = m.group("ts")
            if not ts_str.startswith(day):
                # When reading current log (unrotated), we need to filter by day
                continue
            level = m.group("level")
            msg = m.group("message")
            vid = None
            content = msg
            vm = MSG_VIDEO_RE.match(msg)
            if vm:
                vid = vm.group("video")
                content = vm.group("content")
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            entries.append({"ts": ts, "level": level, "video": vid, "content": content})
    return entries


def collect_metrics(entries: List[Dict]) -> Dict:
    """Aggregate metrics per day and per video."""
    status_per_video: Dict[str, str] = {}
    full_time_per_video: Dict[str, float] = {}
    md_time_per_video: Dict[str, float] = {}
    raw_events_per_video: Dict[str, int] = {}
    md_time_seconds: List[float] = []
    gate_counts = {"up": 0, "down": 0}
    gate_direction_per_video: Dict[str, str] = {}
    first_seen_ts_per_video: Dict[str, datetime] = {}
    # Away/back reaction events and latest reaction state per video
    away_back_events: List[Dict] = []
    reaction_state_per_video: Dict[str, Optional[str]] = {}
    # ReID results per video
    reid_best_score_per_video: Dict[str, float] = {}
    reid_neg_score_per_video: Dict[str, float] = {}
    reid_delta_per_video: Dict[str, float] = {}
    reid_matched_per_video: Dict[str, bool] = {}

    for e in entries:
        content = e["content"]
        vid = e.get("video")
        if vid:
            # Prefer any explicit Status: ... update; fall back to MD-specific pattern
            m = STATUS_ANY_RE.search(content) or STATUS_RE.search(content)
            if m:
                status_per_video[vid] = m.group("status")

            m = FULL_TIME_RE.search(content)
            if m:
                try:
                    full_time_per_video[vid] = float(m.group("seconds"))
                except ValueError:
                    pass

            # First seen timestamp from "New file detected"
            if NEW_FILE_RE.search(content) and vid not in first_seen_ts_per_video:
                first_seen_ts_per_video[vid] = e["ts"]

            m = RAW_EVENTS_RE.search(content)
            if m:
                try:
                    raw_events_per_video[vid] = int(m.group("count"))
                except ValueError:
                    pass

            # Track reaction events for away/back and reset on removal
            if AWAY_RE.search(content):
                away_back_events.append({"type": "away", "video": vid, "ts": e["ts"]})
                reaction_state_per_video[vid] = "away"
            elif BACK_RE.search(content):
                away_back_events.append({"type": "back", "video": vid, "ts": e["ts"]})
                reaction_state_per_video[vid] = "back"
            elif REACTION_REMOVED_RE.search(content):
                # Remove current reaction state for this video
                reaction_state_per_video[vid] = None

            # ReID result parsing
            m = REID_RESULT_RE_NEW.search(content) or REID_RESULT_RE_OLD.search(content)
            if m:
                try:
                    matched = True if m.group("matched") == "True" else False
                    # Prefer new fields if present
                    if m.re is REID_RESULT_RE_NEW:
                        pos = float(m.group("pos"))
                        neg = float(m.group("neg"))
                        delta = float(m.group("delta"))
                        reid_best_score_per_video[vid] = pos
                        reid_neg_score_per_video[vid] = neg
                        reid_delta_per_video[vid] = delta
                    else:
                        score = float(m.group("score"))
                        reid_best_score_per_video[vid] = score
                    reid_matched_per_video[vid] = matched
                except Exception:
                    pass

            # If a later error-level line appears for this video, treat it as status=error
            level = e.get("level")
            if level in ("ERROR", "CRITICAL"):
                status_per_video[vid] = "error"

        # Motion detection time (prefer per video if available)
        m = MD_TIME_RE.search(content)
        if m:
            try:
                secs = float(m.group("seconds"))
                if vid:
                    md_time_per_video[vid] = secs
                else:
                    md_time_seconds.append(secs)
            except ValueError:
                pass

        m = GATE_RE.search(content)
        if m:
            direction = m.group("dir")
            # Count 'both' as one up and one down
            if direction == "both":
                gate_counts["up"] += 1
                gate_counts["down"] += 1
            else:
                gate_counts[direction] += 1
            # Track direction per video; upgrade to 'both' when needed
            if vid:
                current = gate_direction_per_video.get(vid)
                if direction == "both":
                    gate_direction_per_video[vid] = "both"
                elif current is None:
                    gate_direction_per_video[vid] = direction
                elif current != direction and current != "both":
                    gate_direction_per_video[vid] = "both"

    # Count videos per status
    status_counts: Dict[str, int] = {}
    for s in status_per_video.values():
        status_counts[s] = status_counts.get(s, 0) + 1

    # Full processing time stats
    full_times = list(full_time_per_video.values())
    def stats(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        return {
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
        }

    # Combine global MD times and per-video MD times for summary tile
    md_values_all = md_time_seconds + list(md_time_per_video.values())
    md_stats = stats(md_values_all)
    full_stats = stats(full_times)

    # Prefer motion detection time per video, fall back to full processing time
    processing_time_per_video: Dict[str, float] = {}
    vids = set(status_per_video.keys()) | set(full_time_per_video.keys()) | set(md_time_per_video.keys())
    for v in vids:
        if v in md_time_per_video:
            processing_time_per_video[v] = md_time_per_video[v]
        elif v in full_time_per_video:
            processing_time_per_video[v] = full_time_per_video[v]

    return {
        "status_counts": status_counts,
        "full_processing_stats": full_stats,
        "raw_events_per_video": raw_events_per_video,
        "full_time_per_video": full_time_per_video,
        "md_time_per_video": md_time_per_video,
        "processing_time_per_video": processing_time_per_video,
        "status_per_video": status_per_video,
        "motion_detection_stats": md_stats,
        "gate_counts": gate_counts,
        "gate_direction_per_video": gate_direction_per_video,
        "first_seen_ts_per_video": first_seen_ts_per_video,
        "videos_total": len({v for v in status_per_video.keys()} | {v for v in full_time_per_video.keys()} | {v for v in raw_events_per_video.keys()}),
        "away_back_events": away_back_events,
        # Latest reaction state after processing entire day
        "away_videos": sorted([v for v, st in reaction_state_per_video.items() if st == "away"]),
        "back_videos": sorted([v for v, st in reaction_state_per_video.items() if st == "back"]),
        # ReID metrics
        "reid_best_score_per_video": reid_best_score_per_video,
        "reid_neg_score_per_video": reid_neg_score_per_video,
        "reid_delta_per_video": reid_delta_per_video,
        "reid_matched_per_video": reid_matched_per_video,
    }


def _hhmm_from_video_path(basename: str, fallback_ts: Optional[datetime] = None) -> Optional[str]:
    """Derive HH:MM from the video file path: parent folder name is YYYYMMDDHH (take HH),
    and the first two digits of the filename are minutes. Fallback to provided timestamp if path is unavailable.
    """
    try:
        vp = find_video_path(basename)
        if vp:
            parent = os.path.basename(os.path.dirname(vp))
            # Hour: last two digits from YYYYMMDDHH
            hour = parent[-2:] if len(parent) >= 2 and parent[-2:].isdigit() else None
            fname = os.path.basename(vp)
            mm = re.match(r"^(\d{2})", fname)
            minute = mm.group(1) if mm else None
            if hour and minute:
                return f"{hour}:{minute}"
        if fallback_ts is not None:
            return fallback_ts.strftime("%H:%M")
    except Exception:
        if fallback_ts is not None:
            return fallback_ts.strftime("%H:%M")
    return None


def build_away_intervals(entries: List[Dict]) -> List[Dict[str, Optional[str]]]:
    """Build list of intervals for 'object went away' → 'object came back'.
    Pair events by chronological video time (HH:MM derived from path, fallback to log TS).
    Supports stacked openings: multiple consecutive 'away' events create multiple open intervals;
    each subsequent 'back' closes only the most recent open interval. Returns dicts with keys:
    start, end, dur (duration string like '1h14m'). Missing start/end produce 'dur' as None and
    indicate open intervals only when a counterpart does not exist earlier/later in the day's events.
    """
    # Collect events with derived HH:MM and sortable minute index
    # Store minute index, original timestamp, type, hhmm, and video for stable ordering and cancellations
    collected: List[Tuple[int, datetime, str, str, str]] = []  # (minutes, ts, type, hhmm, video)
    for e in entries:
        vid = e.get("video")
        if not vid:
            continue
        content = e.get("content", "")
        is_away = AWAY_RE.search(content)
        is_back = BACK_RE.search(content)
        is_removed = REACTION_REMOVED_RE.search(content)
        if not (is_away or is_back or is_removed):
            continue
        ts = e.get("ts")
        hhmm = _hhmm_from_video_path(vid, fallback_ts=ts)
        if not hhmm and ts:
            hhmm = ts.strftime("%H:%M")
        if not hhmm:
            continue
        try:
            h_str, m_str = hhmm.split(":", 1)
            minutes = int(h_str) * 60 + int(m_str)
        except Exception:
            # Fallback: skip malformed times
            continue
        if is_removed:
            typ = "remove"
        else:
            typ = "away" if is_away else "back"
        collected.append((minutes, ts or datetime.min, typ, hhmm, vid))

    # Sort by derived minute index to ensure chronological pairing
    collected.sort(key=lambda t: (t[0], t[1]))

    # Minimal per-video last-removal filtering:
    # - If a video's LAST reaction is 'remove' → drop all events for that video
    # - Else, drop events at or before that video's LAST 'remove'
    last_remove_minute: Dict[str, int] = {}
    last_reaction_type: Dict[str, str] = {}
    for minutes, _ts, typ, _hhmm, vid in collected:
        if typ == "remove":
            last_remove_minute[vid] = minutes
        last_reaction_type[vid] = typ

    filtered: List[Tuple[int, datetime, str, str, str]] = []
    for minutes, ts, typ, hhmm, vid in collected:
        if last_reaction_type.get(vid) == "remove":
            continue
        lr_min = last_remove_minute.get(vid)
        if lr_min is not None and minutes <= lr_min:
            continue
        filtered.append((minutes, ts, typ, hhmm, vid))

    # Original global pairing updated to support stacked openings
    filtered.sort(key=lambda t: (t[0], t[1]))
    intervals: List[Dict[str, Optional[str]]] = []
    open_starts: List[str] = []  # stack of open 'away' HH:MM values (LIFO)
    for _minutes, _ts, typ, hhmm, _vid in filtered:
        if typ == "away":
            # Push a new open interval start
            open_starts.append(hhmm)
        elif typ == "back":
            if open_starts:
                # Close the most recent open interval
                start_hhmm = open_starts.pop()
                try:
                    sh, sm = [int(x) for x in start_hhmm.split(":", 1)]
                    eh, em = [int(x) for x in hhmm.split(":", 1)]
                    start_min = sh * 60 + sm
                    end_min = eh * 60 + em
                    if end_min > start_min:
                        total = end_min - start_min
                        dh = total // 60
                        dm = total % 60
                        dur = (f"{dh}h" if dh else "") + (f"{dm}m" if dm or not dh else "")
                        intervals.append({"start": start_hhmm, "end": hhmm, "dur": dur})
                    else:
                        # Invalid or zero-length closure; keep the start for a later valid 'back'
                        open_starts.append(start_hhmm)
                except Exception:
                    # Parsing failed; keep the start for a later valid 'back'
                    open_starts.append(start_hhmm)
            else:
                # Back without prior away in this day → open-start interval
                intervals.append({"start": None, "end": hhmm, "dur": None})
        else:
            # 'remove' already filtered out; no-op for safety
            pass

    # Emit any remaining open intervals as open-ended
    for s in open_starts:
        intervals.append({"start": s, "end": None, "dur": None})

    # Sort intervals chronologically by earliest known time (prefer start, else end)
    def _minutes(hhmm: Optional[str]) -> Optional[int]:
        if not hhmm:
            return None
        try:
            h_str, m_str = hhmm.split(":", 1)
            return int(h_str) * 60 + int(m_str)
        except Exception:
            return None

    def _sort_key(iv: Dict[str, Optional[str]]) -> int:
        s = _minutes(iv.get("start"))
        if s is not None:
            return s
        e = _minutes(iv.get("end"))
        return e if e is not None else 10**9

    intervals.sort(key=_sort_key)

    return intervals


def severity_levels() -> List[str]:
    return ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

# Status → color mapping for chart bars
STATUS_COLORS = {
    "no_motion": "#64748b",            # slate
    "no_significant_motion": "#eb980a", # amber
    "no_person": "#f8e805",             # yellow
    "significant_motion": "#6b52fa",    # blue
    "gate_crossing": "#22c55e",        # green
    "error": "#ef4444",               # red
}

# Severity → color mapping for badges
SEVERITY_COLORS = {
    "DEBUG": "#9ca3af",   # gray
    "INFO": "#3b82f6",    # blue
    "WARNING": "#f59e0b", # amber
    "ERROR": "#ef4444",   # red
    "CRITICAL": "#dc2626",# deep red
}


def _hex_to_rgb01(hex_color: str):
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(ch * 2 for ch in h)
    r = int(h[0:2], 16) / 255.0
    g = int(h[2:4], 16) / 255.0
    b = int(h[4:6], 16) / 255.0
    return r, g, b


def _rgb01_to_hex(r: float, g: float, b: float) -> str:
    r8 = max(0, min(255, int(round(r * 255))))
    g8 = max(0, min(255, int(round(g * 255))))
    b8 = max(0, min(255, int(round(b * 255))))
    return f"#{r8:02x}{g8:02x}{b8:02x}"


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def text_color_on_bg(hex_color: str) -> str:
    """Return '#ffffff' or a dark text color based on background luminance."""
    r, g, b = _hex_to_rgb01(hex_color)
    r_lin = _srgb_to_linear(r)
    g_lin = _srgb_to_linear(g)
    b_lin = _srgb_to_linear(b)
    luminance = 0.2126 * r_lin + 0.7152 * g_lin + 0.0722 * b_lin
    return "#0f1115" if luminance > 0.5 else "#ffffff"


# Precompute text colors for badges
STATUS_TEXT_COLORS = {k: text_color_on_bg(v) for k, v in STATUS_COLORS.items()}
SEVERITY_TEXT_COLORS = {k: text_color_on_bg(v) for k, v in SEVERITY_COLORS.items()}

# ReID chip colors: light blue for matched=True, dark blue for matched=False
REID_COLORS = {
    "true": "#65e0ff",   # light blue 300
    "false": "#1e3a8a",  # dark blue 900
}
REID_TEXT_COLORS = {k: text_color_on_bg(v) for k, v in REID_COLORS.items()}

# Neg chip color (violet) and readable text color
NEG_CHIP_COLOR = "#602085"  # violet
NEG_CHIP_TEXT_COLOR = text_color_on_bg(NEG_CHIP_COLOR)


# (moved helper functions above)


def shade_color_for_status(status: str, video: str) -> str:
    """Return a deterministic shade of the base status color for a video.

    Keeps the base hue from STATUS_COLORS, varies lightness (and saturation slightly)
    using a hash of the video name so videos with the same status get different shades
    but remain in the same color family.
    """
    base_hex = STATUS_COLORS.get(status, "#9fb0ff")
    r, g, b = _hex_to_rgb01(base_hex)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    # Deterministic factor in [0, 1)
    f = (abs(hash(video)) % 1000) / 1000.0
    # Wider lightness band for more shade variety, still readable on dark bg
    # Lightness between ~0.42 and ~0.85
    l2 = 0.42 + 0.43 * f
    # Wider saturation variation around the base value, clamped for usability
    # Multiplier ~[0.60, 1.10] → clamp to [0.35, 1.0]
    s_mult = 0.85 + 0.5 * (f - 0.5)
    s2 = max(0.35, min(1.0, s * s_mult))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l2, s2)
    return _rgb01_to_hex(r2, g2, b2)


# --- Video file lookup ---
VIDEO_INDEX: Dict[str, str] = {}

def find_video_path(basename: str) -> Optional[str]:
    """Find full path for a given video basename in VIDEO_FOLDER recursively.

    Caches results in VIDEO_INDEX. Only returns .mp4 files and prevents traversal.
    """
    try:
        name = os.path.basename(basename)
        if not name.lower().endswith(".mp4"):
            return None
        if name in VIDEO_INDEX and os.path.isfile(VIDEO_INDEX[name]):
            return VIDEO_INDEX[name]
        if VIDEO_FOLDER and os.path.isdir(VIDEO_FOLDER):
            for root, _dirs, files in os.walk(VIDEO_FOLDER):
                if name in files:
                    full = os.path.join(root, name)
                    VIDEO_INDEX[name] = full
                    return full
    except Exception:
        return None
    return None


@app.get("/", response_class=HTMLResponse)
def index():
    days = list_log_files()
    today = date.today().strftime("%Y-%m-%d")
    ordered_days = []
    if today in days:
        ordered_days.append((today, days[today]))
    remaining = sorted([(d, p) for d, p in days.items() if d != today], key=lambda t: t[0], reverse=True)
    ordered_days.extend(remaining)

    tmpl = env.get_template("index.html")
    return tmpl.render(ordered_days=ordered_days, log_dir=LOG_DIR)


@app.get("/today", response_class=HTMLResponse)
def today_view(
    severity: Optional[str] = Query(default=None, description="Filter by severity: INFO/WARNING/ERROR/etc."),
    status: Optional[str] = Query(default=None, description="Comma-separated final video statuses (e.g., gate_crossing,significant_motion)"),
    gate: Optional[str] = Query(default=None, description="Filter by gate crossing direction: up/down"),
    video: Optional[str] = Query(default=None, description="Filter logs by a specific video basename (.mp4)"),
    play: Optional[str] = Query(default=None, description="Basename of video to embed (.mp4)"),
):
    """Render today's log page without redirect.

    If today's log is missing, shows the latest available day.
    """
    days = list_log_files()
    today_str = date.today().strftime("%Y-%m-%d")
    if not days:
        raise HTTPException(status_code=404, detail="No log files available")

    day = today_str if today_str in days else sorted(days.keys())[-1]
    path = days[day]

    entries_all = parse_log_lines(path, day)
    metrics_all = collect_metrics(entries_all)
    away_intervals = build_away_intervals(entries_all)

    entries = entries_all
    if severity:
        entries = [e for e in entries if e["level"] == severity]
    status_list_ctx: List[str] = []
    if status:
        spv = metrics_all["status_per_video"]
        allowed = set([s.strip() for s in status.split(",") if s.strip()])
        status_list_ctx = sorted(list(allowed))
        if allowed:
            entries = [e for e in entries if e.get("video") and spv.get(e["video"]) in allowed]
    if gate in ("up", "down"):
        gpv = metrics_all.get("gate_direction_per_video", {})
        entries = [
            e for e in entries
            if e.get("video") and (gpv.get(e["video"]) == gate or gpv.get(e["video"]) == "both")
        ]
    if video:
        entries = [e for e in entries if e.get("video") == video]

    metrics = collect_metrics(entries)

    spv_all = metrics_all.get("status_per_video", {})
    spv = metrics.get("status_per_video", {})
    def color_for(video: str) -> str:
        st = spv_all.get(video) or spv.get(video) or "unknown"
        return shade_color_for_status(st, video)

    videos_present = {e["video"] for e in entries if e.get("video")}
    video_colors = {v: color_for(v) for v in videos_present}

    ptpv = metrics.get("processing_time_per_video", {})
    fst = metrics.get("first_seen_ts_per_video", {})
    ordered_videos = sorted(ptpv.keys(), key=lambda v: (fst.get(v) or datetime.min, v))
    chart_pairs = [(v, ptpv[v]) for v in ordered_videos]

    # Resolve video source if requested
    video_src = None
    if play:
        vp = find_video_path(play)
        if vp:
            video_src = f"/video/{os.path.basename(vp)}"

    tmpl = env.get_template("day.html")
    return tmpl.render(
        day=day,
        is_today_route=True,
        severity=severity,
        status=status,
        gate=gate,
        video=video,
        status_list=status_list_ctx,
        entries=entries,
        metrics=metrics,
        chart_pairs=chart_pairs,
        levels=severity_levels(),
        statuses=sorted(metrics_all["status_counts"].keys()),
        video_colors=video_colors,
        status_colors=STATUS_COLORS,
        severity_colors=SEVERITY_COLORS,
        status_text_colors=STATUS_TEXT_COLORS,
        severity_text_colors=SEVERITY_TEXT_COLORS,
        reid_colors=REID_COLORS,
        reid_text_colors=REID_TEXT_COLORS,
        neg_chip_color=NEG_CHIP_COLOR,
        neg_chip_text_color=NEG_CHIP_TEXT_COLOR,
        statuses_present=sorted(metrics["status_counts"].keys()),
        total_videos_all=metrics_all.get("videos_total", 0),
        log_dir=LOG_DIR,
        play=play,
        video_src=video_src,
            status_counts_all=metrics_all.get("status_counts", {}),
        away_intervals=away_intervals,
    )


@app.get("/day/{day}", response_class=HTMLResponse)
def day_view(
    day: str,
    severity: Optional[str] = Query(default=None, description="Filter by severity: INFO/WARNING/ERROR/etc."),
    status: Optional[str] = Query(default=None, description="Comma-separated final video statuses (e.g., gate_crossing,significant_motion)"),
    gate: Optional[str] = Query(default=None, description="Filter by gate crossing direction: up/down"),
    video: Optional[str] = Query(default=None, description="Filter logs by a specific video basename (.mp4)"),
    play: Optional[str] = Query(default=None, description="Basename of video to embed (.mp4)"),
):
    days = list_log_files()
    if day not in days:
        raise HTTPException(status_code=404, detail=f"No log file for {day}")
    path = days[day]
    entries_all = parse_log_lines(path, day)
    # Compute metrics first to discover statuses
    metrics_all = collect_metrics(entries_all)
    away_intervals = build_away_intervals(entries_all)

    # Apply filters
    entries = entries_all
    if severity:
        entries = [e for e in entries if e["level"] == severity]
    status_list_ctx: List[str] = []
    if status:
        spv = metrics_all["status_per_video"]
        allowed = set([s.strip() for s in status.split(",") if s.strip()])
        status_list_ctx = sorted(list(allowed))
        if allowed:
            entries = [e for e in entries if e.get("video") and spv.get(e["video"]) in allowed]
    if gate in ("up", "down"):
        gpv = metrics_all.get("gate_direction_per_video", {})
        entries = [
            e for e in entries
            if e.get("video") and (gpv.get(e["video"]) == gate or gpv.get(e["video"]) == "both")
        ]
    if video:
        entries = [e for e in entries if e.get("video") == video]

    metrics = collect_metrics(entries)

    # Build colors per video derived from its status base color with shaded variants
    spv_all = metrics_all.get("status_per_video", {})
    spv = metrics.get("status_per_video", {})
    def color_for(video: str) -> str:
        st = spv_all.get(video) or spv.get(video) or "unknown"
        return shade_color_for_status(st, video)

    videos_present = {e["video"] for e in entries if e.get("video")}
    video_colors = {v: color_for(v) for v in videos_present}
    # Build deterministic chart order: by first-seen timestamp, then name
    ptpv = metrics.get("processing_time_per_video", {})
    fst = metrics.get("first_seen_ts_per_video", {})
    ordered_videos = sorted(ptpv.keys(), key=lambda v: (fst.get(v) or datetime.min, v))
    chart_pairs = [(v, ptpv[v]) for v in ordered_videos]
    # Resolve video source if requested
    video_src = None
    if play:
        vp = find_video_path(play)
        if vp:
            video_src = f"/video/{os.path.basename(vp)}"

    tmpl = env.get_template("day.html")
    return tmpl.render(
        day=day,
        is_today_route=False,
        severity=severity,
        status=status,
        gate=gate,
        video=video,
        status_list=status_list_ctx,
        entries=entries,
        metrics=metrics,
        chart_pairs=chart_pairs,
        levels=severity_levels(),
        statuses=sorted(metrics_all["status_counts"].keys()),
        video_colors=video_colors,
        status_colors=STATUS_COLORS,
        severity_colors=SEVERITY_COLORS,
        status_text_colors=STATUS_TEXT_COLORS,
        severity_text_colors=SEVERITY_TEXT_COLORS,
        reid_colors=REID_COLORS,
        reid_text_colors=REID_TEXT_COLORS,
        neg_chip_color=NEG_CHIP_COLOR,
        neg_chip_text_color=NEG_CHIP_TEXT_COLOR,
        statuses_present=sorted(metrics["status_counts"].keys()),
        total_videos_all=metrics_all.get("videos_total", 0),
        log_dir=LOG_DIR,
        play=play,
        video_src=video_src,
            status_counts_all=metrics_all.get("status_counts", {}),
        away_intervals=away_intervals,
    )


@app.get("/video/{basename}")
def stream_video(basename: str):
    """Serve an mp4 video by basename from VIDEO_FOLDER recursively."""
    path = find_video_path(basename)
    if not path:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")


# Static files (CSS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
