import json
import os
import re
import shutil
import sys
import time
import colorsys
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Tuple

import psutil

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape


# Config: align defaults with main.py
LOG_PATH = os.getenv("LOG_PATH", default="")
LOG_DIR = LOG_PATH if LOG_PATH else os.getcwd()
LOG_BASE = os.getenv("LOG_BASENAME", default="video_processor.log")
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER", default=None)
STATS_CACHE_FILE = os.path.join(LOG_DIR, "stats_cache.json")
STATS_CACHE_MAX_AGE_DAYS = 90

# Patterns used by CustomTimedRotatingFileHandler in main.py
ROTATED_PATTERN = re.compile(r"^(?P<base>.+)_((?P<date>\d{4}-\d{2}-\d{2})).log$")

# Log line pattern: optional [W] prefix (legacy) + "YYYY-MM-DD HH:MM:SS - LEVEL - message"
LINE_RE = re.compile(
    r"^(?P<worker_prefix>\[W\] )?(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (?P<level>[A-Z]+) - (?P<message>.*)$"
)

# Optional video basename in message, with optional [W] worker tag: "[basename] [W] Content..."
MSG_VIDEO_RE = re.compile(r"^\[(?P<video>[^\]]+)\]\s*(?P<worker_inline>\[W\] )?(?P<content>.*)$")

# Metrics message patterns
STATUS_RE = re.compile(r"Motion detection complete\. Status: (?P<status>[a-z_]+)")
# Generic status line matcher (covers non-MD contexts like telegram failures)
STATUS_ANY_RE = re.compile(r"Status:\s*(?P<status>[a-z_]+)")
FULL_TIME_RE = re.compile(r"Full processing took (?P<seconds>[0-9]+\.[0-9]+|[0-9]+) seconds")
RAW_EVENTS_RE = re.compile(r"Found (?P<count>\d+) raw motion event\(s\)")
MD_TIME_RE = re.compile(r"Motion detection took (?P<seconds>[0-9]+\.[0-9]+|[0-9]+) seconds")
GATE_RE = re.compile(
    r"Gate crossing detected! Direction: (?P<dir>up|down|both)"
    r"(?:\. Persons Up: (?P<up>\d+), Down: (?P<down>\d+))?"
)
NEW_FILE_RE = re.compile(r"^New file detected:")
# From main.py: "Queuing motion detection... (motion_queue_depth=..., fast_processing=True/False)"
FAST_PROCESSING_RE = re.compile(r"fast_processing=(?P<val>True|False)")
SAVED_FRAME_RE = re.compile(r"Saved \w+ frame to ")
AWAY_RE = re.compile(r"Reaction detected: object went away")
BACK_RE = re.compile(r"Reaction detected: object came back")
REACTION_REMOVED_RE = re.compile(r"Reaction removed\.")
# ReID-accuracy classification: distinguish AUTO vs manual reactions.
# Manual "Reaction detected:" lines come in several flavors; only the
# "object went away/came back" ones represent a positive crossing claim.
AUTO_REACTION_DETECTED_RE = re.compile(r"^AUTO Reaction detected: object (went away|came back)")
AUTO_REACTION_REMOVED_RE = re.compile(r"^AUTO Reaction removed")
MANUAL_REACTION_DETECTED_RE = re.compile(r"^Reaction detected: object (went away|came back)")
MANUAL_REACTION_REMOVED_RE = re.compile(r"^Reaction removed\.")
# ReID result patterns (old and new)
# Old:  "ReID result: matched=True/False, best_score=0.xxx, threshold=..."
# New:  "ReID result: matched=True/False, pos=0.xxx, neg=0.xxx, delta=0.xxx, thr=0.xxx, margin=0.xxx."
SPEEDTRAP_RE = re.compile(r"SpeedTrap (?P<kmh>\d+) km/h")
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

# Cap very large durations when computing averages to reduce outlier skew
CAP_PROCESSING_SECONDS: float = 300.0


def _capped_avg(values: List[float], cap: float = CAP_PROCESSING_SECONDS) -> Optional[float]:
    if not values:
        return None
    capped_sum = 0.0
    n = 0
    for v in values:
        try:
            vv = float(v)
        except Exception:
            continue
        capped_sum += vv if vv <= cap else cap
        n += 1
    if n <= 0:
        return None
    return capped_sum / n


def _minutes_from_hhmmss(hhmmss: Optional[str]) -> Optional[int]:
    if not hhmmss:
        return None
    try:
        h_str, m_str, _s_str = hhmmss.split(":", 2)
        h = int(h_str)
        m = int(m_str)
        if h < 0 or h > 23 or m < 0 or m > 59:
            return None
        return h * 60 + m
    except Exception:
        return None


def collect_away_back_points(entries: List[Dict]) -> List[Dict[str, object]]:
    """Collect away/back event points with minute-of-day.

    Uses filename-derived HH:MM:SS when possible, falls back to log timestamp.
    Returned items: {type: 'away'|'back', video: str, ts: datetime, hhmmss: str, minute: int}
    """
    # Keep only events that happened after the latest "Reaction removed" for each video.
    # This allows corrected/manual reactions after auto-decline to remain visible.
    last_remove_index_by_video: Dict[str, int] = {}
    for idx, e in enumerate(entries):
        vid = e.get("video")
        if not vid:
            continue
        content = e.get("content", "")
        if REACTION_REMOVED_RE.search(content):
            last_remove_index_by_video[vid] = idx

    points: List[Dict[str, object]] = []
    for idx, e in enumerate(entries):
        vid = e.get("video")
        if not vid:
            continue
        # Skip events that are at/before the latest removal marker for this video.
        if idx <= last_remove_index_by_video.get(vid, -1):
            continue
        content = e.get("content", "")
        is_away = AWAY_RE.search(content)
        is_back = BACK_RE.search(content)
        if not (is_away or is_back):
            continue
        ts = e.get("ts")
        hhmmss = _hhmmss_from_video_path(vid, fallback_ts=ts)
        minute = _minutes_from_hhmmss(hhmmss)
        if minute is None and ts:
            minute = ts.hour * 60 + ts.minute
            hhmmss = ts.strftime("%H:%M:%S")
        if minute is None:
            continue
        points.append(
            {
                "type": "away" if is_away else "back",
                "video": vid,
                "ts": ts,
                "hhmmss": hhmmss,
                "minute": minute,
            }
        )
    return points


def classify_reid_outcomes(entries: List[Dict]) -> Dict:
    """Classify ReID accuracy outcomes for a day's log entries.

    Returns {tp, fp, fn, videos, score_sum, score_count, events}.
    `events` is a per-video list ordered by hhmmss with shape
    {video, hhmmss, kind, score} where kind ∈ {TP, FP, FN, FPFN}.
    Score is the latest ReID positive (pos) score per video. The aggregate
    score sums are over truth=crossed videos only (TP + FPFN + FN).
    Videos with no relevant reaction events are ignored entirely.
    """
    auto_detected: Dict[str, bool] = {}
    auto_removed: Dict[str, bool] = {}
    manual_active: Dict[str, bool] = {}
    reid_pos_per_video: Dict[str, float] = {}
    # Capture first-seen ts per video so _hhmmss_from_video_path has a
    # fallback when the filename has no embedded timestamp.
    first_ts_per_video: Dict[str, datetime] = {}

    for e in entries:
        vid = e.get("video")
        if not vid:
            continue
        if vid not in first_ts_per_video:
            ts = e.get("ts")
            if ts is not None:
                first_ts_per_video[vid] = ts
        content = e.get("content", "")
        if AUTO_REACTION_DETECTED_RE.search(content):
            auto_detected[vid] = True
            auto_removed[vid] = False
            continue
        if AUTO_REACTION_REMOVED_RE.search(content):
            if auto_detected.get(vid):
                auto_removed[vid] = True
            continue
        if MANUAL_REACTION_DETECTED_RE.search(content):
            manual_active[vid] = True
            continue
        if MANUAL_REACTION_REMOVED_RE.search(content):
            if manual_active.get(vid):
                manual_active[vid] = False
            elif auto_detected.get(vid) and not auto_removed.get(vid):
                auto_removed[vid] = True
            continue
        m = REID_RESULT_RE_NEW.search(content)
        if m:
            try:
                reid_pos_per_video[vid] = float(m.group("pos"))
            except Exception:
                pass
            continue
        m = REID_RESULT_RE_OLD.search(content)
        if m:
            try:
                reid_pos_per_video[vid] = float(m.group("score"))
            except Exception:
                pass

    tp = fp = fn = 0
    videos = 0
    score_sum = 0.0
    score_count = 0
    events: List[Dict] = []
    relevant = set(auto_detected.keys()) | set(manual_active.keys())
    for vid in relevant:
        ad = auto_detected.get(vid, False)
        ar = auto_removed.get(vid, False)
        ma = manual_active.get(vid, False)
        kind = None
        truth_crossed = False
        if ad and not ar:
            tp += 1
            kind = "TP"
            truth_crossed = True
        elif ad and ar:
            fp += 1
            if ma:
                fn += 1
                kind = "FPFN"  # auto fired wrongly AND user added manual mark
                truth_crossed = True
            else:
                kind = "FP"
        elif not ad and ma:
            fn += 1
            kind = "FN"
            truth_crossed = True
        if kind is None:
            continue
        videos += 1
        score = reid_pos_per_video.get(vid)
        if truth_crossed and score is not None:
            score_sum += score
            score_count += 1
        hhmmss = _hhmmss_from_video_path(vid, fallback_ts=first_ts_per_video.get(vid)) or ""
        events.append({
            "video": vid,
            "hhmmss": hhmmss,
            "kind": kind,
            "score": score,
        })

    events.sort(key=lambda ev: (ev.get("hhmmss") or "", ev.get("video") or ""))

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "videos": videos,
        "score_sum": score_sum,
        "score_count": score_count,
        "events": events,
    }


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
            is_worker = bool(m.group("worker_prefix"))
            vm = MSG_VIDEO_RE.match(msg)
            if vm:
                vid = vm.group("video")
                if vm.group("worker_inline"):
                    is_worker = True
                content = vm.group("content")
            try:
                ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
            entries.append({"ts": ts, "level": level, "video": vid, "content": content, "worker": is_worker})
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
    gate_persons_up_per_video: Dict[str, int] = {}
    gate_persons_down_per_video: Dict[str, int] = {}
    first_seen_ts_per_video: Dict[str, datetime] = {}
    # Fast-processing mode (per video)
    fast_processing_per_video: Dict[str, bool] = {}
    # Worker-processed (per video)
    worker_per_video: Dict[str, bool] = {}
    # Away/back reaction events and latest reaction state per video
    away_back_events: List[Dict] = []
    reaction_state_per_video: Dict[str, Optional[str]] = {}
    # Saved frames indicator per video
    has_frames_per_video: Dict[str, bool] = {}
    # SpeedTrap max speed per video
    speedtrap_per_video: Dict[str, int] = {}
    # ReID results per video
    reid_best_score_per_video: Dict[str, float] = {}
    reid_neg_score_per_video: Dict[str, float] = {}
    reid_delta_per_video: Dict[str, float] = {}
    reid_matched_per_video: Dict[str, bool] = {}
    # Pipeline errors that occur *after* detect_motion produced a valid status
    # (e.g., analyze_video or telegram send failures). Tracked separately so the
    # real motion status is preserved while still surfacing the incident.
    pipeline_error_per_video: Dict[str, bool] = {}

    for e in entries:
        content = e["content"]
        vid = e.get("video")
        if vid:
            # Prefer any explicit Status: ... update; fall back to MD-specific pattern
            m = STATUS_ANY_RE.search(content) or STATUS_RE.search(content)
            if m:
                new_status = m.group("status")
                prev = status_per_video.get(vid)
                # Don't overwrite a real motion status with "error" from a
                # post-pipeline failure (telegram send errors log "Status: error").
                if new_status == "error" and prev and prev != "error":
                    pipeline_error_per_video[vid] = True
                else:
                    status_per_video[vid] = new_status

            m = FULL_TIME_RE.search(content)
            if m:
                try:
                    full_time_per_video[vid] = float(m.group("seconds"))
                except ValueError:
                    pass

            # First seen timestamp from "New file detected"
            if NEW_FILE_RE.search(content) and vid not in first_seen_ts_per_video:
                first_seen_ts_per_video[vid] = e["ts"]

            # Fast-processing marker from queue log
            m = FAST_PROCESSING_RE.search(content)
            if m and m.group("val") == "True":
                # If it was ever True for this video, treat it as fast-processing.
                fast_processing_per_video[vid] = True

            # Worker-processed: any [W] tagged log line for this video
            if e.get("worker"):
                worker_per_video[vid] = True

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

            # Saved frame indicator
            if SAVED_FRAME_RE.search(content):
                has_frames_per_video[vid] = True

            # SpeedTrap detection
            m = SPEEDTRAP_RE.search(content)
            if m:
                kmh = int(m.group("kmh"))
                prev = speedtrap_per_video.get(vid, 0)
                if kmh > prev:
                    speedtrap_per_video[vid] = kmh

            # Error-level lines: if detect_motion already produced a valid status,
            # keep it and just flag pipeline_error. Otherwise fall back to "error".
            level = e.get("level")
            if level in ("ERROR", "CRITICAL"):
                prev = status_per_video.get(vid)
                if prev and prev != "error":
                    pipeline_error_per_video[vid] = True
                else:
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
            # detect_motion.py logs a single "Gate crossing detected!" summary
            # per call (already aggregated across events). If the same video is
            # processed twice — e.g. restart recovery re-queues a file — the
            # log will contain two matching lines. Overwrite rather than sum,
            # so the latest run's counts/direction win.
            if vid:
                gate_direction_per_video[vid] = direction
                up_s, down_s = m.group("up"), m.group("down")
                if up_s is not None and down_s is not None:
                    gate_persons_up_per_video[vid] = int(up_s)
                    gate_persons_down_per_video[vid] = int(down_s)
            else:
                # No video tag on the line — count globally on the fly.
                if direction == "both":
                    gate_counts["up"] += 1
                    gate_counts["down"] += 1
                else:
                    gate_counts[direction] += 1

    # Aggregate global gate counts from deduped per-video directions so a
    # re-processed video doesn't double-count at the daily-total level.
    for direction in gate_direction_per_video.values():
        if direction == "both":
            gate_counts["up"] += 1
            gate_counts["down"] += 1
        elif direction in ("up", "down"):
            gate_counts[direction] += 1

    # Count videos per status
    status_counts: Dict[str, int] = {}
    for s in status_per_video.values():
        status_counts[s] = status_counts.get(s, 0) + 1

    # Full processing time stats
    full_times = list(full_time_per_video.values())
    def stats(values: List[float]) -> Optional[Dict[str, float]]:
        if not values:
            return None
        # Compute average on capped values to avoid skew from sleep/wake outliers
        capped_sum = 0.0
        for v in values:
            try:
                capped_sum += v if v <= CAP_PROCESSING_SECONDS else CAP_PROCESSING_SECONDS
            except Exception:
                # On parse issues, treat as zero contribution
                pass
        avg_val = capped_sum / len(values)
        return {
            "min": min(values),
            "max": max(values),
            "avg": avg_val,
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
        "gate_persons_up_per_video": gate_persons_up_per_video,
        "gate_persons_down_per_video": gate_persons_down_per_video,
        "first_seen_ts_per_video": first_seen_ts_per_video,
        "fast_processing_per_video": fast_processing_per_video,
        "worker_per_video": worker_per_video,
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
        "has_frames_per_video": has_frames_per_video,
        "speedtrap_per_video": speedtrap_per_video,
        "pipeline_error_per_video": pipeline_error_per_video,
    }


def _hhmm_from_video_path(basename: str, fallback_ts: Optional[datetime] = None) -> Optional[str]:
    """Derive HH:MM quickly from basename.
    - New format: parse 14-digit timestamp in basename → HH:MM
    - Legacy format: basename starts with MMSS or MM M SS S; combine MM from basename with hour from fallback timestamp.
      This avoids disk scans while honoring filename minute.
    """
    try:
        name = os.path.basename(basename)
        # Legacy MM/SS at start
        m = re.match(r"^(?P<mm>\d{2})M(?P<ss>\d{2})S", name)
        if not m:
            m = re.match(r"^(?P<mm>\d{2})(?P<ss>\d{2})", name)
        if m and fallback_ts:
            hour = fallback_ts.strftime('%H')
            mm = m.group('mm')
            return f"{hour}:{mm}"
        # New format: parse datetime from basename
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from path_utils import parse_datetime_from_path
        dt = parse_datetime_from_path(name)
        if dt:
            return dt.strftime('%H:%M')
        return fallback_ts.strftime('%H:%M') if fallback_ts else None
    except Exception:
        return fallback_ts.strftime('%H:%M') if fallback_ts else None


def _hhmmss_from_video_path(basename: str, fallback_ts: Optional[datetime] = None) -> Optional[str]:
    """Derive HH:MM:SS using filename-only parsing when possible; avoid disk scans.
    """
    try:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from path_utils import parse_datetime_from_path
        dt = parse_datetime_from_path(os.path.basename(basename))
        if dt:
            return dt.strftime('%H:%M:%S')
        return fallback_ts.strftime('%H:%M:%S') if fallback_ts else None
    except Exception:
        return fallback_ts.strftime('%H:%M:%S') if fallback_ts else None


def _mmMssS_from_video_path(basename: str, fallback_ts: Optional[datetime] = None) -> Optional[str]:
    """Derive <MM>M<SS>S compact label.
    Prefer parsing minutes/seconds directly from basename for legacy files (e.g., 05M04S_... or 0504_...).
    For new format with 14-digit timestamp, use parsed datetime for MM/SS.
    Fall back to provided timestamp when parsing fails.
    """
    try:
        name = os.path.basename(basename)
        # Try legacy patterns at start of basename
        m = re.match(r"^(?P<mm>\d{2})M(?P<ss>\d{2})S", name)
        if not m:
            m = re.match(r"^(?P<mm>\d{2})(?P<ss>\d{2})", name)
        if m:
            mm = m.group("mm")
            ss = m.group("ss")
            return f"{mm}M{ss}S"
        # Else, attempt to parse new format timestamp from basename
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        if root_dir not in sys.path:
            sys.path.insert(0, root_dir)
        from path_utils import parse_datetime_from_path
        dt = parse_datetime_from_path(name)
        if dt:
            return f"{dt.strftime('%M')}M{dt.strftime('%S')}S"
        # Final fallback to provided timestamp
        if fallback_ts:
            return f"{fallback_ts.strftime('%M')}M{fallback_ts.strftime('%S')}S"
        return None
    except Exception:
        if fallback_ts:
            return f"{fallback_ts.strftime('%M')}M{fallback_ts.strftime('%S')}S"
        return None


def build_away_intervals(entries: List[Dict]) -> List[Dict[str, Optional[str]]]:
    """Build list of intervals for 'object went away' → 'object came back'.
    Pair events by chronological video time (HH:MM derived from path, fallback to log TS).
    If a video has multiple consecutive 'Reaction detected' messages of the same type (e.g., multiple
    'object went away' without a 'Reaction removed' in between), only the latest of that run is considered
    for pairing. This prevents creating stacked open intervals from duplicate messages. Returns dicts with keys:
    start, end, dur (duration string like '1h14m'). Missing start/end produce 'dur' as None and
    indicate open intervals only when a counterpart does not exist earlier/later in the day's events.
    """
    # Collect events with derived HH:MM and sortable minute index.
    # Include log sequence index so "Reaction removed" only cancels strictly older
    # events, even when remove/reaction happen in the same minute.
    collected: List[Tuple[int, datetime, int, str, str, str]] = []  # (minutes, ts, seq, type, hhmm, video)
    for seq, e in enumerate(entries):
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
        # Parse from basename only (fast). For legacy names without hour info,
        # _hhmm_from_video_path falls back to the log timestamp hour.
        hhmm = None
        if not hhmm:
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
        collected.append((minutes, ts or datetime.min, seq, typ, hhmm, vid))

    # Sort by derived minute index to ensure chronological pairing
    collected.sort(key=lambda t: (t[0], t[1], t[2]))

    # Per-video latest removal marker by sequence index.
    last_remove_seq: Dict[str, int] = {}
    for _minutes, _ts, seq, typ, _hhmm, vid in collected:
        if typ == "remove":
            last_remove_seq[vid] = seq

    filtered: List[Tuple[int, datetime, int, str, str, str]] = []
    for minutes, ts, seq, typ, hhmm, vid in collected:
        if seq <= last_remove_seq.get(vid, -1):
            continue
        filtered.append((minutes, ts, seq, typ, hhmm, vid))

    # Collapse consecutive duplicate reaction types per video (keep only the latest in each run)
    events_by_video: Dict[str, List[Tuple[int, datetime, str, str]]] = {}
    for minutes, ts, _seq, typ, hhmm, vid in filtered:
        events_by_video.setdefault(vid, []).append((minutes, ts, typ, hhmm))

    condensed: List[Tuple[int, datetime, str, str, str]] = []
    for vid, evs in events_by_video.items():
        # Ensure per-video chronological order
        evs.sort(key=lambda t: (t[0], t[1]))
        last_typ: Optional[str] = None
        buffer_ev: Optional[Tuple[int, datetime, str, str]] = None
        for minutes, ts, typ, hhmm in evs:
            if typ == last_typ:
                # Replace buffered event with the latest of the same type
                buffer_ev = (minutes, ts, typ, hhmm)
            else:
                # Flush previous buffered event when type changes
                if buffer_ev is not None:
                    m, t, ty, h = buffer_ev
                    condensed.append((m, t, ty, h, vid))
                last_typ = typ
                buffer_ev = (minutes, ts, typ, hhmm)
        # Flush the final buffered event for this video
        if buffer_ev is not None:
            m, t, ty, h = buffer_ev
            condensed.append((m, t, ty, h, vid))

    # Use condensed events for pairing
    filtered = sorted(condensed, key=lambda t: (t[0], t[1]))
    
    # Original global pairing
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


def load_stats_cache() -> dict:
    try:
        with open(STATS_CACHE_FILE, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def save_stats_cache(cache: dict) -> None:
    cutoff = (date.today() - timedelta(days=STATS_CACHE_MAX_AGE_DAYS)).isoformat()
    pruned = {d: v for d, v in cache.items() if d >= cutoff}
    tmp_path = STATS_CACHE_FILE + ".tmp"
    try:
        with open(tmp_path, "w") as f:
            json.dump(pruned, f)
        os.replace(tmp_path, STATS_CACHE_FILE)
    except OSError:
        pass


def compute_day_stats(path: str, d: str) -> Tuple[Dict, List[Dict]]:
    """Parse a single day's log and return (per_day_entry, events_list)."""
    entries = parse_log_lines(path, d)
    metrics = collect_metrics(entries)

    videos_total = int(metrics.get("videos_total") or 0)

    status_counts_raw = metrics.get("status_counts") or {}
    status_counts: Dict[str, int] = {}
    try:
        for k, v in dict(status_counts_raw).items():
            if k is None:
                continue
            try:
                status_counts[str(k)] = int(v or 0)
            except Exception:
                status_counts[str(k)] = 0
    except Exception:
        status_counts = {}

    missing = videos_total - sum(status_counts.values())
    if missing > 0:
        status_counts["unknown"] = int(missing)

    md_values_raw = list((metrics.get("md_time_per_video") or {}).values())
    md_values: List[float] = []
    for v in md_values_raw:
        try:
            if v is None:
                continue
            md_values.append(float(v))
        except Exception:
            continue
    md_avg = _capped_avg(md_values)

    full_values_raw = list((metrics.get("full_time_per_video") or {}).values())
    full_values: List[float] = []
    for v in full_values_raw:
        try:
            if v is None:
                continue
            full_values.append(float(v))
        except Exception:
            continue
    full_avg = _capped_avg(full_values)

    day_entry = {
        "day": d,
        "videos_total": videos_total,
        "status_counts": status_counts,
        "md_avg": md_avg,
        "full_avg": full_avg,
    }

    pts = collect_away_back_points(entries)
    events = [{"type": p["type"], "video": p["video"], "hhmmss": p["hhmmss"], "minute": p["minute"]} for p in pts]

    return day_entry, events


@app.get("/stats", response_class=HTMLResponse)
def stats_view():
    """Aggregated statistics across all available log files."""
    days = list_log_files()
    ordered_days = sorted(days.keys())

    per_day: List[Dict[str, object]] = []
    events_all: List[Dict[str, object]] = []

    statuses_seen = set()

    cache = load_stats_cache()
    today = date.today().isoformat()
    dirty = False

    for d in ordered_days:
        if d != today and d in cache:
            cached = cache[d]
            per_day.append(cached["per_day"])
            for p in cached["events"]:
                p2 = dict(p)
                p2["day"] = d
                events_all.append(p2)
            statuses_seen.update(cached["per_day"]["status_counts"].keys())
            continue

        path = days[d]
        day_entry, events = compute_day_stats(path, d)

        per_day.append(day_entry)
        for p in events:
            p2 = dict(p)
            p2["day"] = d
            events_all.append(p2)
        statuses_seen.update(day_entry["status_counts"].keys())

        if d != today:
            cache[d] = {"per_day": day_entry, "events": events}
            dirty = True

    if dirty:
        save_stats_cache(cache)

    # Events heatmap (alternate mode): for each 15-min bin, count how many distinct
    # days had >=1 event in that bin (separately for away/back).
    start_offset = 6 * 60
    total_minutes = 18 * 60
    bin_minutes = 15
    bins = total_minutes // bin_minutes  # 72
    days_count = len(ordered_days)
    away_bin_days = {}
    back_bin_days = {}
    any_hour_days = {}
    away_hour_days = {}
    back_hour_days = {}
    for p in events_all:
        try:
            minute = int(p.get("minute"))
            day = p.get("day")
            typ = p.get("type")
        except Exception:
            continue

        if day is None:
            continue
        if minute < start_offset or minute >= (start_offset + total_minutes):
            continue

        idx = (minute - start_offset) // bin_minutes
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1

        hour_idx = (minute - start_offset) // 60
        if hour_idx < 0:
            hour_idx = 0
        if hour_idx >= (total_minutes // 60):
            hour_idx = (total_minutes // 60) - 1
        any_hour_days.setdefault(hour_idx, set()).add(day)

        if typ == "away":
            away_bin_days.setdefault(idx, set()).add(day)
            away_hour_days.setdefault(hour_idx, set()).add(day)
        else:
            back_bin_days.setdefault(idx, set()).add(day)
            back_hour_days.setdefault(hour_idx, set()).add(day)

    away_days = [len(away_bin_days.get(i, set())) for i in range(bins)]
    back_days = [len(back_bin_days.get(i, set())) for i in range(bins)]
    max_bin_days = 0
    if away_days:
        max_bin_days = max(max_bin_days, max(away_days))
    if back_days:
        max_bin_days = max(max_bin_days, max(back_days))

    hours = total_minutes // 60  # 18
    hour_days = [len(any_hour_days.get(i, set())) for i in range(hours)]
    hour_away_days = [len(away_hour_days.get(i, set())) for i in range(hours)]
    hour_back_days = [len(back_hour_days.get(i, set())) for i in range(hours)]
    max_hour_days = max(hour_days) if hour_days else 0

    events_heatmap = {
        "start_offset": start_offset,
        "bin_minutes": bin_minutes,
        "bins": bins,
        "days_count": days_count,
        "away_days": away_days,
        "back_days": back_days,
        "max_bin_days": max_bin_days,
        "hours": hours,
        "hour_days": hour_days,
        "hour_away_days": hour_away_days,
        "hour_back_days": hour_back_days,
        "max_hour_days": max_hour_days,
    }

    # Weekday heatmap (% of days per weekday that had an event in the bin)
    # Use hourly bins to keep the visualization compact/readable.
    wd_bin_minutes = 60
    wd_bins = total_minutes // wd_bin_minutes  # 18
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_to_weekday: Dict[str, int] = {}
    weekday_day_counts: List[int] = [0] * 7
    for d in ordered_days:
        try:
            wd = date.fromisoformat(d).weekday()
        except Exception:
            continue
        if wd < 0 or wd > 6:
            continue
        day_to_weekday[d] = wd
        weekday_day_counts[wd] += 1

    away_wd_bin_days = {}
    back_wd_bin_days = {}
    for p in events_all:
        try:
            minute = int(p.get("minute"))
            day = p.get("day")
            typ = p.get("type")
        except Exception:
            continue

        if not day or day not in day_to_weekday:
            continue
        if minute < start_offset or minute >= (start_offset + total_minutes):
            continue

        wd = day_to_weekday[day]
        idx = (minute - start_offset) // wd_bin_minutes
        if idx < 0:
            idx = 0
        if idx >= wd_bins:
            idx = wd_bins - 1

        key = (wd, idx)
        if typ == "away":
            away_wd_bin_days.setdefault(key, set()).add(day)
        else:
            back_wd_bin_days.setdefault(key, set()).add(day)

    away_wd_counts: List[List[int]] = [[0 for _ in range(wd_bins)] for _ in range(7)]
    back_wd_counts: List[List[int]] = [[0 for _ in range(wd_bins)] for _ in range(7)]
    for (wd, idx), ds in away_wd_bin_days.items():
        try:
            away_wd_counts[int(wd)][int(idx)] = len(ds)
        except Exception:
            continue
    for (wd, idx), ds in back_wd_bin_days.items():
        try:
            back_wd_counts[int(wd)][int(idx)] = len(ds)
        except Exception:
            continue

    max_pct_away = 0.0
    max_pct_back = 0.0
    for wd in range(7):
        denom = weekday_day_counts[wd]
        if denom <= 0:
            continue
        for idx in range(wd_bins):
            a = away_wd_counts[wd][idx]
            b = back_wd_counts[wd][idx]
            max_pct_away = max(max_pct_away, float(a) / float(denom))
            max_pct_back = max(max_pct_back, float(b) / float(denom))

    events_weekday_heatmap = {
        "start_offset": start_offset,
        "bin_minutes": wd_bin_minutes,
        "bins": wd_bins,
        "weekday_labels": weekday_labels,
        "weekday_day_counts": weekday_day_counts,
        "away_counts": away_wd_counts,
        "back_counts": back_wd_counts,
        "max_pct_away": max_pct_away,
        "max_pct_back": max_pct_back,
    }

    preferred_statuses = list(STATUS_COLORS.keys())
    other_statuses = sorted(
        [s for s in statuses_seen if s not in preferred_statuses and s != "unknown"]
    )
    status_order = [s for s in preferred_statuses if s in statuses_seen] + other_statuses
    if "unknown" in statuses_seen:
        status_order.append("unknown")

    videos_color = SEVERITY_COLORS.get("INFO", "#3b82f6")
    unknown_color = SEVERITY_COLORS.get("DEBUG", "#9ca3af")

    status_colors: Dict[str, str] = {}
    for s in status_order:
        if s == "unknown":
            status_colors[s] = unknown_color
        else:
            status_colors[s] = STATUS_COLORS.get(s, videos_color)
    status_text_colors: Dict[str, str] = {s: text_color_on_bg(c) for s, c in status_colors.items()}

    tmpl = env.get_template("stats.html")
    return tmpl.render(
        per_day=per_day,
        events=events_all,
        events_heatmap=events_heatmap,
        events_weekday_heatmap=events_weekday_heatmap,
        log_dir=LOG_DIR,
        away_color=STATUS_COLORS.get("gate_crossing", "#22c55e"),
        back_color=STATUS_COLORS.get("error", "#ef4444"),
        videos_color=videos_color,
        md_color=SEVERITY_COLORS.get("DEBUG", "#9ca3af"),
        full_color=SEVERITY_COLORS.get("INFO", "#3b82f6"),
        status_order=status_order,
        status_colors=status_colors,
        status_text_colors=status_text_colors,
    )


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

    # Filename-derived HH:MM:SS per video (fallback to first-seen TS)
    # Build labels for all videos referenced in this page to minimize lookups
    videos_for_labels = set(metrics.get("status_per_video", {}).keys()) | videos_present
    hhmmss_per_video = {v: _hhmmss_from_video_path(v, fallback_ts=fst.get(v)) for v in videos_for_labels}
    # Filename-derived <MM>M<SS>S per video (fallback to first-seen TS)
    mmss_per_video = {v: _mmMssS_from_video_path(v, fallback_ts=fst.get(v)) for v in videos_for_labels}

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
        fast_processing_per_video_all=metrics_all.get("fast_processing_per_video", {}),
        worker_per_video_all=metrics_all.get("worker_per_video", {}),
        away_intervals=away_intervals,
        hhmmss_per_video=hhmmss_per_video,
        mmss_per_video=mmss_per_video,
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
    # Filename-derived HH:MM:SS per video (fallback to first-seen TS)
    videos_for_labels = set(metrics.get("status_per_video", {}).keys()) | videos_present
    hhmmss_per_video = {v: _hhmmss_from_video_path(v, fallback_ts=fst.get(v)) for v in videos_for_labels}
    # Filename-derived <MM>M<SS>S per video (fallback to first-seen TS)
    mmss_per_video = {v: _mmMssS_from_video_path(v, fallback_ts=fst.get(v)) for v in videos_for_labels}
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
        fast_processing_per_video_all=metrics_all.get("fast_processing_per_video", {}),
        worker_per_video_all=metrics_all.get("worker_per_video", {}),
        away_intervals=away_intervals,
        hhmmss_per_video=hhmmss_per_video,
        mmss_per_video=mmss_per_video,
    )


@app.get("/video/{basename}")
def stream_video(basename: str):
    """Serve an mp4 video by basename from VIDEO_FOLDER recursively."""
    path = find_video_path(basename)
    if not path:
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")


# ---------------------------------------------------------------------------
# JSON API endpoints (for Android app)
# ---------------------------------------------------------------------------

# ReID gallery paths (same defaults as telegram_notification.py)
_SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
REID_GALLERY_PATH = os.getenv("REID_GALLERY_PATH", os.path.join(_SCRIPT_DIR, "person_of_interest"))
REID_NEGATIVE_GALLERY_PATH = os.getenv("REID_NEGATIVE_GALLERY_PATH", os.path.join(_SCRIPT_DIR, "person_of_interest_negative"))

# Processing ledger path (same default as main.py)
TEMP_DIR = os.getenv("TEMP_DIR", os.path.join(_SCRIPT_DIR, "temp"))
PROCESSING_LEDGER_PATH = os.path.join(TEMP_DIR, "processing_ledger.json")
REID_METRICS_CACHE_FILE = os.path.join(TEMP_DIR, "reid_metrics_cache.json")
# v2: added score_sum/score_count per day for ReID match-score MA.
# v3: cache per-day events list (video, hhmmss, kind, score) for the event-strip view.
REID_METRICS_CACHE_VERSION = 3

# Worker URL for proxying health checks
WORKER_URL = os.getenv("WORKER_URL", "")

# In-memory per-day cache for parsed log data (max 7 entries, 30s TTL)
_day_cache: Dict[str, Dict[str, object]] = {}
_DAY_CACHE_TTL = 30.0
_DAY_CACHE_MAX = 7

# CPU percent cache (non-blocking reads)
_cpu_cache: Dict[str, object] = {"value": 0.0, "ts": 0.0}
_CPU_CACHE_TTL = 10.0

# Worker health cache
_worker_health_cache: Dict[str, object] = {"data": None, "ts": 0.0}
_WORKER_HEALTH_CACHE_TTL = 30.0
# Timestamp (epoch seconds) when the worker first went offline in the current offline streak.
# Cleared as soon as the worker reports healthy again.
_worker_offline_since: Optional[float] = None

# TEMP_DIR media index cache (avoids repeated directory scans per request)
_TEMP_MEDIA_INDEX_TTL = float(os.getenv("TEMP_MEDIA_INDEX_TTL", "15"))
_temp_media_index_cache: Dict[str, object] = {"data": None, "ts": 0.0}


_EMPTY_METRICS = {
    "status_counts": {}, "status_per_video": {}, "processing_time_per_video": {},
    "gate_direction_per_video": {},
    "gate_persons_up_per_video": {}, "gate_persons_down_per_video": {},
    "first_seen_ts_per_video": {},
    "fast_processing_per_video": {}, "worker_per_video": {},
    "reid_best_score_per_video": {}, "reid_neg_score_per_video": {},
    "reid_delta_per_video": {}, "reid_matched_per_video": {},
    "has_frames_per_video": {}, "speedtrap_per_video": {},
    "pipeline_error_per_video": {},
    "away_videos": [], "back_videos": [], "away_back_events": [],
    "videos_total": 0, "full_processing_stats": None, "motion_detection_stats": None,
    "gate_counts": {"up": 0, "down": 0}, "raw_events_per_video": {},
    "full_time_per_video": {}, "md_time_per_video": {},
}


def _get_day_parsed(day_str: Optional[str] = None) -> Tuple[str, List[Dict], Dict]:
    """Return (day, entries, metrics) for the given day (or today), with per-day caching."""
    now = time.time()
    days = list_log_files()
    if day_str and day_str in days:
        day = day_str
    else:
        today_str = date.today().strftime("%Y-%m-%d")
        day = today_str if today_str in days else (sorted(days.keys())[-1] if days else None)

    if not day or day not in days:
        return (day_str or date.today().strftime("%Y-%m-%d"), [], dict(_EMPTY_METRICS))

    # Check cache
    cached = _day_cache.get(day)
    if cached and now - cached["ts"] < _DAY_CACHE_TTL:
        return cached["data"]

    path = days[day]
    entries = parse_log_lines(path, day)
    metrics = collect_metrics(entries)
    result = (day, entries, metrics)

    # Store in cache, evict oldest if full
    _day_cache[day] = {"data": result, "ts": now}
    if len(_day_cache) > _DAY_CACHE_MAX:
        oldest_key = min(_day_cache, key=lambda k: _day_cache[k]["ts"])
        del _day_cache[oldest_key]

    return result


def _fetch_worker_health_sync(timeout: float = 3.0) -> Dict:
    """Fetch worker health. Returns status dict or raises on failure."""
    import urllib.request
    req = urllib.request.Request(f"{WORKER_URL}/health", method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _store_worker_health(data: Dict) -> Dict:
    """Update cache + offline_since tracking. Returns `data` possibly annotated with offline_since."""
    global _worker_offline_since
    now = time.time()
    is_offline = data.get("status") != "ok"
    if is_offline:
        if _worker_offline_since is None:
            _worker_offline_since = now
        # Stamp the response with the offline_since time for downstream consumers.
        data = dict(data)
        data["offline_since"] = datetime.fromtimestamp(_worker_offline_since).strftime("%H:%M:%S")
    else:
        _worker_offline_since = None
    _worker_health_cache["data"] = data
    _worker_health_cache["ts"] = now
    return data


def _fetch_worker_health_bg():
    """Background thread: fetch worker health and update cache."""
    try:
        data = _fetch_worker_health_sync()
        _store_worker_health(data)
    except Exception:
        _store_worker_health({"status": "offline"})
        try:
            from worker.client import try_wol_if_needed
            try_wol_if_needed()
        except Exception:
            pass


def _get_worker_health() -> Optional[Dict]:
    """Return cached worker health. Uses a short sync attempt first, falls back to background."""
    if not WORKER_URL:
        return None
    now = time.time()
    if now - _worker_health_cache["ts"] < _WORKER_HEALTH_CACHE_TTL:
        return _worker_health_cache["data"]
    # Try a quick synchronous fetch (covers the online-worker case)
    try:
        data = _fetch_worker_health_sync(timeout=0.5)
        return _store_worker_health(data)
    except Exception:
        pass
    # Worker didn't respond quickly — do the full retry in background
    import threading
    threading.Thread(target=_fetch_worker_health_bg, daemon=True).start()
    # Return stale cache if available, otherwise record a fresh offline state.
    if _worker_health_cache["data"] is not None:
        return _worker_health_cache["data"]
    return _store_worker_health({"status": "offline"})


@app.get("/api/days")
def api_days():
    """Return list of available log days."""
    days = sorted(list_log_files().keys())
    return {"days": days}


@app.get("/api/today/videos")
def api_today_videos(day: Optional[str] = Query(default=None)):
    day, entries, metrics = _get_day_parsed(day)
    spv = metrics.get("status_per_video", {})
    ptpv = metrics.get("processing_time_per_video", {})
    gpv = metrics.get("gate_direction_per_video", {})
    fst = metrics.get("first_seen_ts_per_video", {})
    fpv = metrics.get("fast_processing_per_video", {})
    wpv = metrics.get("worker_per_video", {})
    rspv = metrics.get("reid_best_score_per_video", {})
    rnpv = metrics.get("reid_neg_score_per_video", {})
    rmpv = metrics.get("reid_matched_per_video", {})
    hfpv = metrics.get("has_frames_per_video", {})
    stpv = metrics.get("speedtrap_per_video", {})
    pepv = metrics.get("pipeline_error_per_video", {})
    away_videos = set(metrics.get("away_videos", []))
    back_videos = set(metrics.get("back_videos", []))

    all_vids = set(spv.keys()) | set(ptpv.keys())
    videos = []
    for v in sorted(all_vids, key=lambda x: (fst.get(x) or datetime.min, x)):
        t = _hhmmss_from_video_path(v, fallback_ts=fst.get(v))
        away_back = None
        if v in away_videos:
            away_back = "away"
        elif v in back_videos:
            away_back = "back"
        videos.append({
            "basename": v,
            "time": t,
            "status": spv.get(v),
            "gate_direction": gpv.get(v),
            "processing_time_s": round(ptpv[v], 2) if v in ptpv else None,
            "raw_events": metrics.get("raw_events_per_video", {}).get(v),
            "fast_processing": fpv.get(v, False),
            "worker": wpv.get(v, False),
            "reid_matched": rmpv.get(v),
            "reid_score": round(rspv[v], 3) if v in rspv else None,
            "reid_neg": round(rnpv[v], 3) if v in rnpv else None,
            "away_back": away_back,
            "has_frames": hfpv.get(v, False),
            "speed_kmh": stpv.get(v),
            "pipeline_error": pepv.get(v, False),
        })
    return {"day": day, "videos": videos}


@app.get("/api/today/video/{basename}/logs")
def api_today_video_logs(basename: str, severity: Optional[str] = Query(default=None), day: Optional[str] = Query(default=None)):
    day, entries, _metrics = _get_day_parsed(day)
    filtered = [e for e in entries if e.get("video") == basename]
    if severity:
        filtered = [e for e in filtered if e["level"] == severity]
    return {
        "basename": basename,
        "entries": [
            {
                "ts": e["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                "level": e["level"],
                "content": e["content"],
                "worker": e.get("worker", False),
            }
            for e in filtered
        ],
    }


@app.get("/api/today/video/{basename}/reid-crops")
def api_today_video_reid_crops(basename: str):
    stem = os.path.splitext(basename)[0]
    idx = _get_temp_media_index()
    names = idx.get("reid_by_stem", {}).get(stem, [])
    crops = [f"/api/image/{fname}" for fname in names]
    return {"basename": basename, "crops": crops}


@app.get("/api/today/video/{basename}/frames")
def api_today_video_frames(basename: str):
    """Return insignificant/no_person frame image URLs for a video."""
    stem = os.path.splitext(basename)[0]
    idx = _get_temp_media_index()
    names = idx.get("frames_by_stem", {}).get(stem, [])
    frames = [f"/api/image/{fname}" for fname in names]
    return {"basename": basename, "frames": frames}


@app.get("/api/today/video/{basename}/pose")
def api_today_video_pose(basename: str):
    """Return pose clip URLs for a video."""
    stem = os.path.splitext(basename)[0]
    idx = _get_temp_media_index()
    names = idx.get("pose_by_stem", {}).get(stem, [])
    clips = [f"/api/highlight/{fname}" for fname in names]
    return {"basename": basename, "clips": clips}


_DAILY_DIR_RE = re.compile(r"^\d{8}$")


def _walk_daily_dirs(base_dir: str):
    """Yield (root, files) only for YYYYMMDD daily subdirectories of base_dir."""
    try:
        for entry in os.scandir(base_dir):
            if entry.is_dir() and _DAILY_DIR_RE.match(entry.name):
                try:
                    files = os.listdir(entry.path)
                    yield entry.path, files
                except OSError:
                    continue
    except OSError:
        return


_FRAME_NAME_RE = re.compile(r"^\d{2}H(?P<stem>.+)_(?:insignificant|no_person)_\d+\.(?:jpg|jpeg|png)$", re.IGNORECASE)


def _extract_frame_stem(fname: str) -> Optional[str]:
    m = _FRAME_NAME_RE.match(fname)
    if not m:
        return None
    return m.group("stem")


def _build_temp_media_index() -> Dict[str, object]:
    data: Dict[str, object] = {
        "image_paths": {},          # basename -> abs path (TEMP_DIR daily dirs)
        "video_paths": {},          # basename -> abs path (TEMP_DIR daily dirs)
        "reid_by_stem": {},         # video stem -> [image basenames]
        "frames_by_stem": {},       # video stem -> [image basenames]
        "pose_by_stem": {},         # video stem -> [pose clip basenames]
        "highlight_names": set(),   # set[highlight clip basename]
    }

    try:
        daily_dirs = []
        for entry in os.scandir(TEMP_DIR):
            if entry.is_dir() and _DAILY_DIR_RE.match(entry.name):
                daily_dirs.append(entry.path)
        # Prefer newest daily folders on basename collisions.
        daily_dirs.sort(reverse=True)
    except OSError:
        daily_dirs = []

    for dpath in daily_dirs:
        try:
            files = os.listdir(dpath)
        except OSError:
            continue

        for fname in files:
            lower = fname.lower()
            fpath = os.path.join(dpath, fname)

            if lower.endswith((".jpg", ".jpeg", ".png")):
                data["image_paths"].setdefault(fname, fpath)
                if "_reid_best" in fname:
                    stem = fname.split("_reid_best", 1)[0]
                    data["reid_by_stem"].setdefault(stem, []).append(fname)
                else:
                    stem = _extract_frame_stem(fname)
                    if stem:
                        data["frames_by_stem"].setdefault(stem, []).append(fname)
                continue

            if lower.endswith(".mp4"):
                data["video_paths"].setdefault(fname, fpath)
                if "_pose_" in fname:
                    stem = fname.split("_pose_", 1)[0]
                    data["pose_by_stem"].setdefault(stem, []).append(fname)
                else:
                    data["highlight_names"].add(fname)

    for key in ("reid_by_stem", "frames_by_stem", "pose_by_stem"):
        for stem, names in data[key].items():
            names.sort()

    return data


def _get_temp_media_index() -> Dict[str, object]:
    now = time.time()
    cached = _temp_media_index_cache.get("data")
    if cached is not None and (now - float(_temp_media_index_cache.get("ts", 0.0))) < _TEMP_MEDIA_INDEX_TTL:
        return cached
    data = _build_temp_media_index()
    _temp_media_index_cache["data"] = data
    _temp_media_index_cache["ts"] = now
    return data


@app.get("/api/today/video/{basename}/highlight")
def api_today_video_highlight(basename: str):
    """Check if a highlight clip exists for this video and return its URL."""
    name = os.path.basename(basename)
    if not name.lower().endswith(".mp4"):
        return {"basename": basename, "highlight_url": None}
    idx = _get_temp_media_index()
    if name in idx.get("highlight_names", set()):
        return {"basename": basename, "highlight_url": f"/api/highlight/{name}"}
    return {"basename": basename, "highlight_url": None}


@app.get("/api/today/video/{basename}/full")
def api_today_video_full(basename: str):
    """Return full-video URL only when the source file currently exists."""
    name = os.path.basename(basename)
    if not name.lower().endswith(".mp4"):
        return {"basename": basename, "video_url": None}
    path = find_video_path(name)
    if path and os.path.isfile(path):
        return {"basename": basename, "video_url": f"/video/{name}"}
    return {"basename": basename, "video_url": None}


@app.get("/api/highlight/{basename}")
def api_serve_highlight(basename: str):
    """Serve a highlight clip by basename from TEMP_DIR daily subdirectories."""
    name = os.path.basename(basename)
    if not name.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only mp4 files allowed")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    idx = _get_temp_media_index()
    path = idx.get("video_paths", {}).get(name)
    if path and os.path.isfile(path):
        return FileResponse(path, media_type="video/mp4")
    raise HTTPException(status_code=404, detail="Highlight not found")


@app.get("/api/today/gate-crossings")
def api_today_gate_crossings(day: Optional[str] = Query(default=None)):
    """Return all videos that have ReID crops, with crop URLs pre-fetched."""
    day_str, entries, metrics = _get_day_parsed(day)
    spv = metrics.get("status_per_video", {})
    gpv = metrics.get("gate_direction_per_video", {})
    gup = metrics.get("gate_persons_up_per_video", {})
    gdn = metrics.get("gate_persons_down_per_video", {})
    fst = metrics.get("first_seen_ts_per_video", {})
    rspv = metrics.get("reid_best_score_per_video", {})
    rmpv = metrics.get("reid_matched_per_video", {})
    rnpv = metrics.get("reid_neg_score_per_video", {})

    idx = _get_temp_media_index()
    reid_by_stem = idx.get("reid_by_stem", {})

    away_videos = set(metrics.get("away_videos", []))
    back_videos = set(metrics.get("back_videos", []))

    all_vids = set(spv.keys()) | set(fst.keys())
    gate_crossings = []
    for v in sorted(all_vids, key=lambda x: (fst.get(x) or datetime.min, x), reverse=True):
        stem = os.path.splitext(v)[0]
        crops = [f"/api/image/{fname}" for fname in reid_by_stem.get(stem, [])]
        if not crops:
            continue
        t = _hhmmss_from_video_path(v, fallback_ts=fst.get(v))
        away_back = None
        if v in away_videos:
            away_back = "away"
        elif v in back_videos:
            away_back = "back"
        gate_crossings.append({
            "basename": v,
            "time": t,
            "direction": gpv.get(v),
            "persons_up": gup.get(v),
            "persons_down": gdn.get(v),
            "status": spv.get(v),
            "reid_matched": rmpv.get(v),
            "reid_score": round(rspv[v], 3) if v in rspv else None,
            "reid_neg": round(rnpv[v], 3) if v in rnpv else None,
            "away_back": away_back,
            "crops": crops,
        })

    return {"day": day_str, "gate_crossings": gate_crossings}


def _fill_ongoing_dur(intervals: List[Dict], day_iso: str) -> List[Dict]:
    """For today's open away intervals (start known, end missing), compute the
    elapsed duration as `now - start` and write it into `dur` so the UI can
    show e.g. "1h 24m" alongside the start time.
    """
    if day_iso != date.today().isoformat():
        return intervals
    now = datetime.now()
    now_minute = now.hour * 60 + now.minute
    out: List[Dict] = []
    for it in intervals:
        if it.get("start") and not it.get("end") and not it.get("dur"):
            try:
                sh, sm = [int(x) for x in it["start"].split(":", 1)]
                start_minute = sh * 60 + sm
                delta = now_minute - start_minute
                if delta > 0:
                    dh = delta // 60
                    dm = delta % 60
                    dur = (f"{dh}h" if dh else "") + (f"{dm}m" if dm or not dh else "")
                    out.append({**it, "dur": dur, "ongoing": True})
                    continue
            except Exception:
                pass
        out.append(it)
    return out


@app.get("/api/today/stats")
def api_today_stats(day: Optional[str] = Query(default=None)):
    day, entries, metrics = _get_day_parsed(day)
    away_intervals = _fill_ongoing_dur(build_away_intervals(entries), day)

    fst = metrics.get("first_seen_ts_per_video", {})
    ptpv = metrics.get("processing_time_per_video", {})
    spv = metrics.get("status_per_video", {})
    ordered_videos = sorted(ptpv.keys(), key=lambda v: (fst.get(v) or datetime.min, v))

    chart = []
    for v in ordered_videos:
        t = _hhmmss_from_video_path(v, fallback_ts=fst.get(v))
        chart.append({
            "basename": v,
            "time": t,
            "seconds": round(ptpv[v], 2),
            "status": spv.get(v),
        })

    md_stats = metrics.get("motion_detection_stats")
    full_stats = metrics.get("full_processing_stats")

    processing_stats = {}
    if md_stats:
        processing_stats["md_min"] = round(md_stats["min"], 2)
        processing_stats["md_max"] = round(md_stats["max"], 2)
        processing_stats["md_avg"] = round(md_stats["avg"], 2)
    if full_stats:
        processing_stats["full_min"] = round(full_stats["min"], 2)
        processing_stats["full_max"] = round(full_stats["max"], 2)
        processing_stats["full_avg"] = round(full_stats["avg"], 2)

    return {
        "day": day,
        "videos_total": metrics.get("videos_total", 0),
        "status_counts": metrics.get("status_counts", {}),
        "gate_counts": metrics.get("gate_counts", {"up": 0, "down": 0}),
        "processing_stats": processing_stats,
        "processing_chart": chart,
        "away_intervals": away_intervals,
    }


@app.get("/api/stats/overall")
def api_stats_overall():
    days = list_log_files()
    ordered_days = sorted(days.keys())

    per_day: List[Dict] = []
    events_all: List[Dict] = []
    cache = load_stats_cache()
    today = date.today().isoformat()
    dirty = False

    for d in ordered_days:
        if d != today and d in cache:
            cached = cache[d]
            per_day.append(cached["per_day"])
            for p in cached["events"]:
                p2 = dict(p)
                p2["day"] = d
                events_all.append(p2)
            continue

        path = days[d]
        day_entry, events = compute_day_stats(path, d)
        per_day.append(day_entry)
        for p in events:
            p2 = dict(p)
            p2["day"] = d
            events_all.append(p2)

        if d != today:
            cache[d] = {"per_day": day_entry, "events": events}
            dirty = True

    # Extend events_all with events from cached days outside the log-retention
    # window so weekday-pattern stats (heatmaps) match the prediction history.
    # Per-day video counts above stay log-only as the user prefers.
    for cd, cd_data in cache.items():
        if cd == today or cd in days:
            continue
        for p in cd_data.get("events", []):
            p2 = dict(p)
            p2["day"] = cd
            events_all.append(p2)

    if dirty:
        save_stats_cache(cache)

    # Build heatmaps (same logic as stats_view but return JSON).
    # Pattern denominator = past days (today excluded since it's only partial).
    pattern_days = sorted({d for d in ordered_days if d != today} | {d for d in cache.keys() if d != today})
    start_offset = 6 * 60
    total_minutes = 18 * 60
    bin_minutes = 15
    bins = total_minutes // bin_minutes
    days_count = len(pattern_days)

    away_bin_days: Dict[int, set] = {}
    back_bin_days: Dict[int, set] = {}
    for p in events_all:
        try:
            minute = int(p.get("minute"))
            d = p.get("day")
            typ = p.get("type")
        except Exception:
            continue
        if d == today:
            continue
        if d is None or minute < start_offset or minute >= (start_offset + total_minutes):
            continue
        idx = max(0, min(bins - 1, (minute - start_offset) // bin_minutes))
        if typ == "away":
            away_bin_days.setdefault(idx, set()).add(d)
        else:
            back_bin_days.setdefault(idx, set()).add(d)

    events_heatmap = {
        "start_offset": start_offset,
        "bin_minutes": bin_minutes,
        "bins": bins,
        "days_count": days_count,
        "away_days": [len(away_bin_days.get(i, set())) for i in range(bins)],
        "back_days": [len(back_bin_days.get(i, set())) for i in range(bins)],
    }

    # Weekday heatmap — also over the pattern (cache-extended, today excluded)
    # window so the percentages match what the prediction code computes.
    wd_bin_minutes = 60
    wd_bins = total_minutes // wd_bin_minutes
    weekday_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_to_weekday: Dict[str, int] = {}
    weekday_day_counts = [0] * 7
    for d in pattern_days:
        try:
            wd = date.fromisoformat(d).weekday()
            day_to_weekday[d] = wd
            weekday_day_counts[wd] += 1
        except Exception:
            continue

    away_wd: Dict[tuple, set] = {}
    back_wd: Dict[tuple, set] = {}
    # Per-cell occurrence lists (all events, not deduped by day) so the UI can
    # show the actual dates+times that landed in a selected weekday-hour bin.
    away_wd_events: Dict[tuple, List[Dict[str, str]]] = {}
    back_wd_events: Dict[tuple, List[Dict[str, str]]] = {}
    for p in events_all:
        try:
            minute = int(p.get("minute"))
            d = p.get("day")
            typ = p.get("type")
        except Exception:
            continue
        if not d or d not in day_to_weekday:
            continue
        if minute < start_offset or minute >= (start_offset + total_minutes):
            continue
        wd = day_to_weekday[d]
        idx = max(0, min(wd_bins - 1, (minute - start_offset) // wd_bin_minutes))
        key = (wd, idx)
        hhmmss = p.get("hhmmss") or ""
        occ = {"date": d, "hhmmss": str(hhmmss)}
        if typ == "away":
            away_wd.setdefault(key, set()).add(d)
            away_wd_events.setdefault(key, []).append(occ)
        else:
            back_wd.setdefault(key, set()).add(d)
            back_wd_events.setdefault(key, []).append(occ)

    away_counts = [[len(away_wd.get((wd, idx), set())) for idx in range(wd_bins)] for wd in range(7)]
    back_counts = [[len(back_wd.get((wd, idx), set())) for idx in range(wd_bins)] for wd in range(7)]

    def _sorted_occurrences(bag: Dict[tuple, List[Dict[str, str]]]) -> List[List[List[Dict[str, str]]]]:
        # Most recent first (date desc, then time desc).
        return [
            [
                sorted(
                    bag.get((wd, idx), []),
                    key=lambda o: (o.get("date", ""), o.get("hhmmss", "")),
                    reverse=True,
                )
                for idx in range(wd_bins)
            ]
            for wd in range(7)
        ]

    weekday_heatmap = {
        "start_offset": start_offset,
        "bin_minutes": wd_bin_minutes,
        "bins": wd_bins,
        "weekday_labels": weekday_labels,
        "weekday_day_counts": weekday_day_counts,
        "away_counts": away_counts,
        "back_counts": back_counts,
        "away_occurrences": _sorted_occurrences(away_wd_events),
        "back_occurrences": _sorted_occurrences(back_wd_events),
    }

    return {
        "per_day": per_day,
        "events_heatmap": events_heatmap,
        "weekday_heatmap": weekday_heatmap,
    }


def load_reid_metrics_cache() -> dict:
    """Load the ReID accuracy cache. Returns {"version": N, "days": {date: {tp,fp,fn,videos}}}."""
    try:
        with open(REID_METRICS_CACHE_FILE, "r") as f:
            data = json.load(f)
        if not isinstance(data, dict) or data.get("version") != REID_METRICS_CACHE_VERSION:
            return {"version": REID_METRICS_CACHE_VERSION, "days": {}}
        days = data.get("days")
        if not isinstance(days, dict):
            return {"version": REID_METRICS_CACHE_VERSION, "days": {}}
        return data
    except (OSError, json.JSONDecodeError, ValueError):
        return {"version": REID_METRICS_CACHE_VERSION, "days": {}}


def save_reid_metrics_cache(cache: dict) -> None:
    tmp_path = REID_METRICS_CACHE_FILE + ".tmp"
    try:
        os.makedirs(os.path.dirname(REID_METRICS_CACHE_FILE), exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(cache, f)
        os.replace(tmp_path, REID_METRICS_CACHE_FILE)
    except OSError:
        pass


def _reid_metrics_from_counts(tp: int, fp: int, fn: int) -> Dict[str, Optional[float]]:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {"precision": precision, "recall": recall, "f1": f1}


@app.get("/api/stats/reid")
def api_stats_reid():
    """ReID auto-detection accuracy metrics per day with totals and 7-day MA.

    Classification rules: see classify_reid_outcomes(). Closed days are cached
    on disk under TEMP_DIR; today is always recomputed live so late corrections
    are reflected immediately.
    """
    days = list_log_files()
    today = date.today().isoformat()

    cache = load_reid_metrics_cache()
    cached_days = cache["days"]
    dirty = False

    # Surface the union of (cached days, days with log files) so the chart
    # retains history past the log-retention window. The cache is the source
    # of truth for closed days; logs only matter for today and for first-time
    # population of newly-closed days.
    all_days = sorted(set(cached_days.keys()) | set(days.keys()))

    per_day: List[Dict] = []
    # Parallel list of raw count dicts so the MA below can aggregate score_sum/
    # score_count correctly (averaging averages would skew it).
    per_day_raw: List[Dict] = []
    for d in all_days:
        if d == today:
            # Today is always live. If today has no log file yet, skip it.
            if d not in days:
                continue
            entries = parse_log_lines(days[d], d)
            counts = classify_reid_outcomes(entries)
        elif d in cached_days:
            c = cached_days[d]
            counts = {
                "tp": int(c.get("tp", 0)),
                "fp": int(c.get("fp", 0)),
                "fn": int(c.get("fn", 0)),
                "videos": int(c.get("videos", 0)),
                "score_sum": float(c.get("score_sum", 0.0)),
                "score_count": int(c.get("score_count", 0)),
                "events": list(c.get("events", [])),
            }
        elif d in days:
            # Closed day not yet cached — parse the log and store it.
            entries = parse_log_lines(days[d], d)
            counts = classify_reid_outcomes(entries)
            cached_days[d] = counts
            dirty = True
        else:
            # Can't happen — d came from the union of both sets.
            continue

        metrics = _reid_metrics_from_counts(counts["tp"], counts["fp"], counts["fn"])
        score_avg = (counts["score_sum"] / counts["score_count"]) if counts["score_count"] > 0 else None
        # Attach a crop URL to each event at response time (not cached) so the
        # URL reflects the current state of TEMP_DIR media — if a crop has been
        # cleaned up, the field is just null.
        media_idx = _get_temp_media_index()
        reid_by_stem = media_idx.get("reid_by_stem", {}) or {}
        decorated_events: List[Dict] = []
        for ev in counts.get("events", []):
            stem = os.path.splitext(ev.get("video", ""))[0]
            crops = reid_by_stem.get(stem, [])
            crop_url = f"/api/image/{crops[0]}" if crops else None
            decorated_events.append({**ev, "crop_url": crop_url})

        per_day.append({
            "date": d,
            "tp": counts["tp"],
            "fp": counts["fp"],
            "fn": counts["fn"],
            "videos": counts["videos"],
            "score_avg": score_avg,
            "events": decorated_events,
            **metrics,
        })
        per_day_raw.append(counts)

    if dirty:
        save_reid_metrics_cache(cache)

    # Totals across the whole period
    total_tp = sum(p["tp"] for p in per_day)
    total_fp = sum(p["fp"] for p in per_day)
    total_fn = sum(p["fn"] for p in per_day)
    total_videos = sum(p["videos"] for p in per_day)
    total_score_sum = sum(r["score_sum"] for r in per_day_raw)
    total_score_count = sum(r["score_count"] for r in per_day_raw)
    totals = {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "videos": total_videos,
        "score_avg": (total_score_sum / total_score_count) if total_score_count > 0 else None,
        **_reid_metrics_from_counts(total_tp, total_fp, total_fn),
    }

    # 7-day moving average over raw counts (not over ratios — that doesn't
    # average correctly when sample sizes vary). Same principle for the score
    # MA: aggregate score_sum / score_count over the window.
    moving_avg: List[Dict] = []
    window = 7
    for i, p in enumerate(per_day):
        lo = max(0, i - window + 1)
        win_raw = per_day_raw[lo:i + 1]
        wtp = sum(w["tp"] for w in win_raw)
        wfp = sum(w["fp"] for w in win_raw)
        wfn = sum(w["fn"] for w in win_raw)
        wss = sum(w["score_sum"] for w in win_raw)
        wsc = sum(w["score_count"] for w in win_raw)
        moving_avg.append({
            "date": p["date"],
            "score_avg": (wss / wsc) if wsc > 0 else None,
            **_reid_metrics_from_counts(wtp, wfp, wfn),
        })

    return {
        "totals": totals,
        "per_day": per_day,
        "moving_avg_7d": moving_avg,
    }


@app.get("/api/monitoring")
def api_monitoring():
    # Master stats
    now = time.time()
    if now - _cpu_cache["ts"] > _CPU_CACHE_TTL:
        _cpu_cache["value"] = psutil.cpu_percent(interval=None)
        _cpu_cache["ts"] = now

    mem = psutil.virtual_memory()
    bat = psutil.sensors_battery()
    master = {
        "cpu_percent": _cpu_cache["value"],
        "memory_percent": mem.percent,
        "memory_used_mb": mem.used // (1024 * 1024),
        "memory_total_mb": mem.total // (1024 * 1024),
        "battery_percent": bat.percent if bat else None,
        "battery_plugged": bat.power_plugged if bat else None,
        "battery_time_left_s": bat.secsleft if bat and bat.secsleft > 0 else None,
    }

    # Worker stats (proxied with caching)
    worker = _get_worker_health()

    # Recent processing ledger entries
    ledger_recent = []
    try:
        with open(PROCESSING_LEDGER_PATH, "r") as f:
            ledger = json.load(f)
        items = sorted(
            ledger.items(),
            key=lambda x: x[1].get("end_ts") or x[1].get("start_ts") or x[1].get("detected_ts") or 0,
            reverse=True,
        )[:10]
        for k, v in items:
            ledger_recent.append({
                "file": os.path.basename(k),
                "status": v.get("status"),
                "detected_ts": v.get("detected_ts"),
                "start_ts": v.get("start_ts"),
                "end_ts": v.get("end_ts"),
                "fast_processing": v.get("fast_processing"),
                "telegram_status": v.get("telegram_status"),
            })
    except (OSError, json.JSONDecodeError, ValueError):
        pass

    # Tesla SoC: try today's log first, fall back to cache file
    tesla = None
    tesla_soc_file = os.path.join(TEMP_DIR, "tesla_soc.txt")
    try:
        day, entries, _metrics = _get_day_parsed()
        tesla_re = re.compile(r"Tesla SoC fetched: (?P<pct>\d+)%")
        # Find the latest Tesla SoC log entry
        for e in reversed(entries):
            m = tesla_re.search(e.get("content", ""))
            if m:
                tesla = {
                    "battery_percent": int(m.group("pct")),
                    "fetched_ts": e["ts"].strftime("%Y-%m-%d %H:%M:%S"),
                }
                break
    except Exception:
        pass
    if tesla is None:
        try:
            with open(tesla_soc_file, "r") as f:
                soc_val = f.read().strip()
            if soc_val:
                mtime = os.path.getmtime(tesla_soc_file)
                tesla = {
                    "battery_percent": int(soc_val),
                    "fetched_ts": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S"),
                }
        except (OSError, ValueError):
            pass

    return {"master": master, "worker": worker, "tesla": tesla, "ledger_recent": ledger_recent}


# ---------------------------------------------------------------------------
# Next-event prediction
# ---------------------------------------------------------------------------
# Uses the same hourly bin structure as the weekday heatmap (06:00–24:00 in
# 60-minute bins). For the current weekday, walks remaining bins forward from
# "now" and picks the first whose historical hit rate clears a threshold; if
# none does, tries 2-bin rolling windows; otherwise no prediction.

_PRED_BIN_MINUTES = 60
_PRED_START_OFFSET = 6 * 60
_PRED_TOTAL_MINUTES = 18 * 60
_PRED_BINS = _PRED_TOTAL_MINUTES // _PRED_BIN_MINUTES
_PRED_CONFIDENCE_THRESHOLD = 0.30
_PRED_MIN_WEEKDAY_SAMPLES = 4
_PRED_BACK_MIN_AFTER_AWAY_MIN = 20
_PRED_MINUTE_ROUNDING = 10


def _aggregate_weekday_events(stats_cache: dict, target_weekday: int, exclude_day: str):
    """Walk the overall-stats cache and aggregate per-bin events for a weekday.

    Returns a dict with `weekday_days_total` and two bin-keyed dicts of
    {days: set[str], minutes: list[int]} for away and back.
    """
    away_bins: Dict[int, Dict] = {}
    back_bins: Dict[int, Dict] = {}
    weekday_days: set = set()

    for d, day_data in stats_cache.items():
        if d == exclude_day:
            continue
        try:
            wd = date.fromisoformat(d).weekday()
        except Exception:
            continue
        if wd != target_weekday:
            continue
        weekday_days.add(d)
        for ev in day_data.get("events", []):
            try:
                minute = int(ev.get("minute"))
                typ = ev.get("type")
            except Exception:
                continue
            if minute < _PRED_START_OFFSET or minute >= _PRED_START_OFFSET + _PRED_TOTAL_MINUTES:
                continue
            bin_idx = (minute - _PRED_START_OFFSET) // _PRED_BIN_MINUTES
            target = away_bins if typ == "away" else (back_bins if typ == "back" else None)
            if target is None:
                continue
            slot = target.setdefault(bin_idx, {"days": set(), "minutes": []})
            slot["days"].add(d)
            slot["minutes"].append(minute)

    return {
        "weekday_days_total": len(weekday_days),
        "away_bins": away_bins,
        "back_bins": back_bins,
    }


def _round_minute_to_nearest(minute: int, step: int) -> int:
    return int(round(minute / step) * step)


def _format_predicted_hhmm(minute: int) -> str:
    minute = max(0, min(24 * 60 - 1, minute))
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _build_prediction_result(
    kind: str,
    predicted_minute: int,
    confidence: float,
    basis_count: int,
    basis_total: int,
    now_minute: int,
    imminent: bool = False,
) -> Dict:
    rounded = _round_minute_to_nearest(predicted_minute, _PRED_MINUTE_ROUNDING)
    # If a non-imminent prediction would still land in the past, push it to
    # the next round-multiple of the rounding step.
    if not imminent and rounded < now_minute:
        rounded = _round_minute_to_nearest(now_minute + _PRED_MINUTE_ROUNDING, _PRED_MINUTE_ROUNDING)
    return {
        "kind": kind,                                          # "away" | "back"
        "predicted_hhmm": _format_predicted_hhmm(rounded),
        "confidence": round(confidence, 3),
        "basis_count": basis_count,
        "basis_total": basis_total,
        "imminent": imminent,
    }


def compute_next_prediction(
    now: datetime,
    state: str,
    away_start_minute: Optional[int],
    stats_cache: dict,
) -> Optional[Dict]:
    """Return a prediction dict or None.

    state: "home" → predict next "away"; "away" → predict next "back".
    For back predictions, candidate times within `_PRED_BACK_MIN_AFTER_AWAY_MIN`
    of the away start are skipped (people don't usually return that fast).
    """
    if state not in ("home", "away"):
        return None

    weekday = now.weekday()
    today_iso = now.date().isoformat()
    now_minute = now.hour * 60 + now.minute

    agg = _aggregate_weekday_events(stats_cache, weekday, exclude_day=today_iso)
    weekday_total = agg["weekday_days_total"]
    if weekday_total < _PRED_MIN_WEEKDAY_SAMPLES:
        return None

    bins = agg["away_bins"] if state == "home" else agg["back_bins"]
    kind = "away" if state == "home" else "back"

    current_bin = max(0, (now_minute - _PRED_START_OFFSET) // _PRED_BIN_MINUTES)
    current_bin = min(current_bin, _PRED_BINS - 1)

    def _bin_mean(bin_data: Dict) -> int:
        ms = bin_data.get("minutes") or []
        return sum(ms) // len(ms) if ms else 0

    def _passes_back_gap(minute: int) -> bool:
        if state != "away" or away_start_minute is None:
            return True
        return minute >= away_start_minute + _PRED_BACK_MIN_AFTER_AWAY_MIN

    # Pass 1: single-bin, threshold = 30%.
    for bin_idx in range(current_bin, _PRED_BINS):
        bin_data = bins.get(bin_idx)
        if not bin_data:
            continue
        unique_days = len(bin_data.get("days") or set())
        if unique_days == 0:
            continue
        confidence = unique_days / weekday_total
        if confidence < _PRED_CONFIDENCE_THRESHOLD:
            continue
        mean_minute = _bin_mean(bin_data)
        if not _passes_back_gap(mean_minute):
            continue
        imminent = bin_idx == current_bin and mean_minute < now_minute
        return _build_prediction_result(
            kind=kind,
            predicted_minute=mean_minute,
            confidence=confidence,
            basis_count=unique_days,
            basis_total=weekday_total,
            now_minute=now_minute,
            imminent=imminent,
        )

    # Pass 2: 2-bin rolling window. Union unique days across the pair so we
    # don't overcount a day that has events in both bins.
    for bin_idx in range(current_bin, _PRED_BINS - 1):
        a = bins.get(bin_idx) or {"days": set(), "minutes": []}
        b = bins.get(bin_idx + 1) or {"days": set(), "minutes": []}
        union_days = (a.get("days") or set()) | (b.get("days") or set())
        if not union_days:
            continue
        confidence = len(union_days) / weekday_total
        if confidence < _PRED_CONFIDENCE_THRESHOLD:
            continue
        all_minutes = (a.get("minutes") or []) + (b.get("minutes") or [])
        if not all_minutes:
            continue
        mean_minute = sum(all_minutes) // len(all_minutes)
        if not _passes_back_gap(mean_minute):
            continue
        imminent = bin_idx == current_bin and mean_minute < now_minute
        return _build_prediction_result(
            kind=kind,
            predicted_minute=mean_minute,
            confidence=confidence,
            basis_count=len(union_days),
            basis_total=weekday_total,
            now_minute=now_minute,
            imminent=imminent,
        )

    return None


@app.get("/api/events/latest")
def api_events_latest(since: Optional[str] = Query(default=None, description="ISO timestamp or HH:MM:SS")):
    day, entries, _metrics = _get_day_parsed()
    points = collect_away_back_points(entries)

    since_dt = None
    if since:
        try:
            since_dt = datetime.strptime(since, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            try:
                since_dt = datetime.strptime(f"{day} {since}", "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    since_dt = datetime.strptime(f"{day} {since}", "%Y-%m-%d %H:%M")
                except ValueError:
                    pass

    events = []
    for p in points:
        if since_dt and p["ts"] and p["ts"] <= since_dt:
            continue
        events.append({
            "type": p["type"],
            "video": p["video"],
            "ts": p["ts"].strftime("%Y-%m-%d %H:%M:%S") if p["ts"] else None,
            "hhmmss": p["hhmmss"],
        })

    # Determine current home/away status from the latest event by real-world event time
    # (not log/processing time — videos can be processed out of order when there's a backlog).
    current_status = None
    current_status_since = None
    away_start_minute: Optional[int] = None
    if points:
        latest = max(points, key=lambda p: (p.get("minute") or -1, p.get("ts") or datetime.min))
        current_status = "home" if latest["type"] == "back" else "away"
        current_status_since = latest["hhmmss"]
        if current_status == "away":
            try:
                away_start_minute = int(latest.get("minute"))
            except Exception:
                away_start_minute = None

    # Predict next away/back from same-weekday history. Effective state defaults
    # to "home" when there are no events today yet — predicts the first leave.
    effective_state = current_status or "home"
    try:
        prediction = compute_next_prediction(
            now=datetime.now(),
            state=effective_state,
            away_start_minute=away_start_minute,
            stats_cache=load_stats_cache(),
        )
    except Exception:
        prediction = None

    return {
        "events": events,
        "current_status": current_status,
        "current_status_since": current_status_since,
        "next_prediction": prediction,
        "server_ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/image/{basename}")
def api_serve_image(basename: str):
    """Serve a ReID crop image by basename from TEMP_DIR or VIDEO_FOLDER."""
    name = os.path.basename(basename)
    if not name.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only image files allowed")
    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    idx = _get_temp_media_index()
    temp_path = idx.get("image_paths", {}).get(name)
    if temp_path and os.path.isfile(temp_path):
        return FileResponse(temp_path, media_type="image/jpeg")

    # Fallback path for uncommon cases where images are under VIDEO_FOLDER.
    if VIDEO_FOLDER:
        try:
            for root, _dirs, files in os.walk(VIDEO_FOLDER):
                if name in files:
                    return FileResponse(os.path.join(root, name), media_type="image/jpeg")
        except OSError:
            pass
    raise HTTPException(status_code=404, detail="Image not found")


from pydantic import BaseModel


class ReidCopyRequest(BaseModel):
    image_path: str
    target: str  # "positive" or "negative"


@app.post("/api/reid/copy")
def api_reid_copy(req: ReidCopyRequest):
    """Copy a ReID crop image to the positive or negative gallery."""
    if req.target not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="target must be 'positive' or 'negative'")

    # Resolve source: basename only, find in VIDEO_FOLDER
    src_name = os.path.basename(req.image_path)
    if ".." in src_name or "/" in src_name or "\\" in src_name:
        raise HTTPException(status_code=400, detail="Invalid filename")

    idx = _get_temp_media_index()
    src_path = idx.get("image_paths", {}).get(src_name)
    if (not src_path) and VIDEO_FOLDER:
        try:
            for root, _dirs, files in os.walk(VIDEO_FOLDER):
                if src_name in files:
                    src_path = os.path.join(root, src_name)
                    break
        except OSError:
            src_path = None
    if not src_path:
        raise HTTPException(status_code=404, detail="Source image not found")

    gallery = REID_GALLERY_PATH if req.target == "positive" else REID_NEGATIVE_GALLERY_PATH
    os.makedirs(gallery, exist_ok=True)
    dst = os.path.join(gallery, src_name)
    try:
        shutil.copy2(src_path, dst)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Copy failed: {e}")

    return {"status": "ok", "destination": dst}


def _resolve_gallery_path(target: str, filename: str) -> str:
    """Validate target + filename and return the absolute gallery file path.

    Rejects path traversal and unexpected separators so only files directly
    inside the configured gallery dirs can be addressed.
    """
    if target not in ("positive", "negative"):
        raise HTTPException(status_code=400, detail="target must be 'positive' or 'negative'")
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=400, detail="Only image files allowed")
    safe_name = os.path.basename(filename)
    if safe_name != filename or ".." in safe_name or "/" in safe_name or "\\" in safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename")
    gallery = REID_GALLERY_PATH if target == "positive" else REID_NEGATIVE_GALLERY_PATH
    return os.path.join(gallery, safe_name)


@app.get("/api/gallery/{target}/{filename}")
def api_serve_gallery_image(target: str, filename: str):
    """Serve a gallery reference crop by {target}/{filename}."""
    path = _resolve_gallery_path(target, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Gallery crop not found")
    return FileResponse(path, media_type="image/jpeg")


@app.delete("/api/gallery/{target}/{filename}")
def api_delete_gallery_image(target: str, filename: str):
    """Delete a gallery reference crop. The person_id gallery cache is keyed
    by file signature (name+size+mtime), so the next ReID run auto-rebuilds
    the embeddings — no explicit cache invalidation needed here."""
    path = _resolve_gallery_path(target, filename)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Gallery crop not found")
    try:
        os.remove(path)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {e}")
    return {"status": "ok", "deleted": os.path.basename(path)}


# Static files (CSS)
static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
