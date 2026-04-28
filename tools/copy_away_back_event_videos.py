#!/usr/bin/env python3
"""Copy away/back event videos using the log dashboard API.

This script queries:
  - /api/events/latest
and downloads matching video files from:
  - /video/{basename}

Example:
  python tools/copy_away_back_event_videos.py \
    --base-url http://192.168.1.33:8192 \
    --output-dir temp/pose_signature_candidates
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, List
from urllib.parse import quote
from urllib.request import urlopen


AWAY_TOKEN = "Reaction detected: object went away"
BACK_TOKEN = "Reaction detected: object came back"


def _normalize_base_url(raw: str) -> str:
    base = (raw or "").strip().rstrip("/")
    if not base.startswith("http://") and not base.startswith("https://"):
        base = "http://" + base
    return base


def _fetch_json(url: str, timeout: float = 15.0) -> Dict:
    with urlopen(url, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload)


def _download_file(url: str, dst_path: str, timeout: float = 45.0) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with urlopen(url, timeout=timeout) as resp:
        content = resp.read()
    with open(dst_path, "wb") as f:
        f.write(content)


def _event_sort_key(event: Dict) -> str:
    # API provides ts like "YYYY-MM-DD HH:MM:SS" or None.
    ts = event.get("ts")
    if isinstance(ts, str):
        return ts
    return ""


def _parse_since_day(since: str | None) -> str | None:
    if not since:
        return None
    s = str(since).strip()
    if not s:
        return None
    # Full datetime format used by API docs.
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return None


def _scan_events_from_videos_state(base_url: str, since_day: str | None = None) -> List[Dict]:
    """Fast fallback: use /api/today/videos for each day and collect away_back flags."""
    out: List[Dict] = []

    days_url = f"{base_url}/api/days"
    days_payload = _fetch_json(days_url)
    days = days_payload.get("days") or []
    if not isinstance(days, list):
        return out

    days_sorted = [d for d in days if isinstance(d, str) and d]
    days_sorted.sort()
    if since_day:
        days_sorted = [d for d in days_sorted if d >= since_day]

    print(f"INFO: fallback scan via /api/today/videos across {len(days_sorted)} day(s).")

    for day in days_sorted:
        print(f"INFO: scanning day {day}...")
        if not isinstance(day, str) or not day:
            continue

        videos_url = f"{base_url}/api/today/videos?day={quote(day)}"
        videos_payload = _fetch_json(videos_url)
        videos = videos_payload.get("videos") or []
        if not isinstance(videos, list):
            continue

        for v in videos:
            if not isinstance(v, dict):
                continue
            basename = v.get("basename")
            if not isinstance(basename, str) or not basename.lower().endswith(".mp4"):
                continue
            away_back = str(v.get("away_back") or "").lower()
            if away_back in ("away", "back"):
                out.append(
                    {
                        "type": away_back,
                        "video": basename,
                        "ts": None,
                        "source_day": day,
                        "source": "videos_state",
                    }
                )

    return out


def _scan_events_from_logs(base_url: str, since_day: str | None = None, max_days: int | None = None) -> List[Dict]:
    """Deep fallback: scan per-video logs and extract away/back lines."""
    out: List[Dict] = []

    days_url = f"{base_url}/api/days"
    days_payload = _fetch_json(days_url)
    days = days_payload.get("days") or []
    if not isinstance(days, list):
        return out

    days_sorted = [d for d in days if isinstance(d, str) and d]
    days_sorted.sort()
    if since_day:
        days_sorted = [d for d in days_sorted if d >= since_day]
    if max_days and max_days > 0:
        days_sorted = days_sorted[-max_days:]

    print(f"INFO: deep fallback scan via per-video logs across {len(days_sorted)} day(s).")

    for day in days_sorted:
        print(f"INFO: deep-scanning day {day}...")
        if not isinstance(day, str) or not day:
            continue

        videos_url = f"{base_url}/api/today/videos?day={quote(day)}"
        videos_payload = _fetch_json(videos_url)
        videos = videos_payload.get("videos") or []
        if not isinstance(videos, list):
            continue

        for v in videos:
            if not isinstance(v, dict):
                continue
            basename = v.get("basename")
            if not isinstance(basename, str) or not basename.lower().endswith(".mp4"):
                continue

            logs_url = f"{base_url}/api/today/video/{quote(basename)}/logs?day={quote(day)}"
            logs_payload = _fetch_json(logs_url)
            entries = logs_payload.get("entries") or []
            if not isinstance(entries, list):
                continue

            for e in entries:
                if not isinstance(e, dict):
                    continue
                content = str(e.get("content") or "")
                ts = e.get("ts")
                if AWAY_TOKEN in content:
                    out.append({"type": "away", "video": basename, "ts": ts, "source_day": day, "source": "logs_scan"})
                elif BACK_TOKEN in content:
                    out.append({"type": "back", "video": basename, "ts": ts, "source_day": day, "source": "logs_scan"})

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy away/back event videos from dashboard API.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8192",
        help="Dashboard base URL (default: http://127.0.0.1:8192)",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join("temp", "pose_signature_candidates"),
        help="Directory where matching videos will be saved.",
    )
    parser.add_argument(
        "--event-type",
        choices=["away", "back", "both"],
        default="both",
        help="Which event videos to copy (default: both).",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Optional since filter passed to /api/events/latest (YYYY-MM-DD HH:MM:SS or HH:MM[:SS]).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite destination files if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print actions and write manifest; do not download files.",
    )
    parser.add_argument(
        "--deep-log-scan",
        action="store_true",
        help="If fast fallback is empty, run deep per-video log scan (slower).",
    )
    parser.add_argument(
        "--max-fallback-days",
        type=int,
        default=14,
        help="Limit days scanned in fallback mode (default: 14).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = _normalize_base_url(args.base_url)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    events_url = f"{base_url}/api/events/latest"
    if args.since:
        events_url = f"{events_url}?since={quote(args.since)}"
    since_day = _parse_since_day(args.since)

    source_mode = "events_latest"
    try:
        data = _fetch_json(events_url)
    except Exception as exc:
        print(f"ERROR: failed to query events API: {events_url}")
        print(f"DETAIL: {exc}")
        return 1

    events = data.get("events") or []
    if not isinstance(events, list):
        print("ERROR: API response has invalid 'events' format.")
        return 1

    if len(events) == 0:
        try:
            events = _scan_events_from_videos_state(base_url, since_day=since_day)
            source_mode = "videos_state_fallback"
            print("INFO: /api/events/latest returned no events; used /api/today/videos fallback.")
            if len(events) == 0 and args.deep_log_scan:
                events = _scan_events_from_logs(
                    base_url,
                    since_day=since_day,
                    max_days=max(1, int(args.max_fallback_days)),
                )
                source_mode = "logs_scan_fallback"
                print("INFO: fast fallback returned no events; used deep per-video logs fallback.")
        except Exception as exc:
            print("WARNING: fallback log scan failed; continuing with empty event list.")
            print(f"DETAIL: {exc}")

    allowed_types = {"away", "back"}
    if args.event_type in {"away", "back"}:
        allowed_types = {args.event_type}

    filtered: List[Dict] = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ev_type = str(e.get("type") or "").lower()
        video = e.get("video")
        if ev_type not in allowed_types:
            continue
        if not isinstance(video, str) or not video.lower().endswith(".mp4"):
            continue
        filtered.append(e)

    filtered.sort(key=_event_sort_key)

    # Keep unique basenames while preserving event order.
    unique_videos: List[str] = []
    seen = set()
    for e in filtered:
        v = e["video"]
        if v in seen:
            continue
        seen.add(v)
        unique_videos.append(v)

    downloaded: List[str] = []
    skipped_existing: List[str] = []
    failed: List[Dict[str, str]] = []

    for basename in unique_videos:
        src_url = f"{base_url}/video/{quote(basename)}"
        dst_path = os.path.join(output_dir, basename)

        if os.path.exists(dst_path) and not args.overwrite:
            skipped_existing.append(basename)
            continue

        if args.dry_run:
            print(f"DRY RUN: {src_url} -> {dst_path}")
            downloaded.append(basename)
            continue

        try:
            _download_file(src_url, dst_path)
            downloaded.append(basename)
            print(f"OK: {basename}")
        except Exception as exc:
            failed.append({"video": basename, "error": str(exc)})
            print(f"FAIL: {basename} ({exc})")

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base_url": base_url,
        "events_url": events_url,
        "source_mode": source_mode,
        "event_type": args.event_type,
        "events_total": len(events),
        "events_filtered": len(filtered),
        "videos_unique": len(unique_videos),
        "downloaded": downloaded,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "events": filtered,
    }

    manifest_path = os.path.join(output_dir, "away_back_events_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("---")
    print(f"Manifest: {manifest_path}")
    print(f"Filtered events: {len(filtered)}")
    print(f"Unique videos: {len(unique_videos)}")
    print(f"Downloaded: {len(downloaded)}")
    print(f"Skipped existing: {len(skipped_existing)}")
    print(f"Failed: {len(failed)}")

    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
