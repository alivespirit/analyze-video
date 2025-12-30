#!/usr/bin/env python3
"""
Replay motion detection on videos listed in a prior day's log,
using a specified detection model, and record new statuses.

- Parses a log file for lines like:
  "[<filename>.mp4] New file detected: <timestamp>/<filename>.mp4"
  and final results:
  "[<filename>.mp4] Motion detection complete. Status: <status>"

- Skips files where prior Status == no_motion.
- Runs detect_motion on remaining files, writing outputs to <temp_dir>/<timestamp>/.
- Records timestamp/filename -> status (from detect_motion result) to a CSV.

Usage:
  python tools/validate_log_replay.py \
    --log-file /path/to/previous_day.log \
    --video-dir /path/to/videos/root \
    --temp-dir /path/to/temp/output \
    --model-path /path/to/new_model_dir

Notes:
- This script sets OBJECT_DETECTION_MODEL_PATH before importing detect_motion.
- detect_motion loads the model at import time, so the env must be set first.
"""

import argparse
import csv
import logging
import os
import re
import sys
import gc
import multiprocessing as mp
from pathlib import Path

# ---------------------------------------------------------------
# CLI + logging setup
# ---------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Replay detect_motion on videos from a log")
    p.add_argument("--log-file", required=True, help="Path to the previous day's log file")
    p.add_argument("--video-dir", required=True, help="Root directory where videos are stored")
    p.add_argument("--temp-dir", required=True, help="Root temp directory to write outputs")
    p.add_argument("--model-path", required=False, help="Path to object detection model to use")
    p.add_argument("--results-file", required=False, default="validation_results.csv", help="CSV filename for results mapping (created inside temp-dir)")
    p.add_argument("--isolate", action="store_true", help="Run each video in a subprocess to fully reclaim memory after processing")
    return p.parse_args()


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    return logging.getLogger("validate_log_replay")


# ---------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------

NEW_FILE_RE = re.compile(r"\[(?P<file>[^\]]+)]\s+New\s+file\s+detected:\s+(?P<ts>\d+)/(?P<name>[^\s]+)")
FINAL_STATUS_RE = re.compile(r"\[(?P<file>[^\]]+)]\s+Motion\s+detection\s+complete\.\s+Status:\s+(?P<status>\w+)")


def parse_log(log_path: Path):
    """Parse the log and return two dicts:
    - files: filename -> timestamp (from 'New file detected')
    - statuses: filename -> last final status (from 'Motion detection complete. Status: ...')
    """
    files = {}
    statuses = {}

    with log_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m_new = NEW_FILE_RE.search(line)
            if m_new:
                filename_bracket = m_new.group("file")  # e.g., 09M27S_1767020967.mp4
                ts = m_new.group("ts")                  # e.g., 2025122917
                name = m_new.group("name")              # e.g., 09M27S_1767020967.mp4
                # Prefer canonical filename from bracket; should match name
                files[filename_bracket] = ts
                continue

            m_status = FINAL_STATUS_RE.search(line)
            if m_status:
                filename_bracket = m_status.group("file")
                status = m_status.group("status")
                statuses[filename_bracket] = status
                continue

    return files, statuses


# ---------------------------------------------------------------
# Subprocess worker for isolated per-video processing
# ---------------------------------------------------------------

def _worker(video_path: str, output_dir: str, model_path: str, repo_root: str, result_queue):
    try:
        if model_path:
            os.environ["OBJECT_DETECTION_MODEL_PATH"] = model_path
        # Ensure repo root on sys.path for child
        if repo_root:
            sys.path.insert(0, repo_root)
        from detect_motion import detect_motion
        result = detect_motion(video_path, output_dir)
        status = result.get("status", "unknown")
        result_queue.put({"status": status})
    except Exception as e:
        result_queue.put({"error": str(e)})


# ---------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------

def main():
    args = parse_args()
    logger = configure_logging()

    log_path = Path(args.log_file)
    video_root = Path(args.video_dir)
    temp_root = Path(args.temp_dir)
    temp_root.mkdir(parents=True, exist_ok=True)

    # Set model path BEFORE importing detect_motion (model loads at import time)
    if args.model_path:
        os.environ["OBJECT_DETECTION_MODEL_PATH"] = args.model_path
        logger.info(f"Using model path: {args.model_path}")

    # Repo root for imports
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    # Import detect_motion only when not isolating per-video
    detect_motion_fn = None
    if not args.isolate:
        from detect_motion import detect_motion
        detect_motion_fn = detect_motion

    files, statuses = parse_log(log_path)
    logger.info(f"Parsed {len(files)} 'New file detected' entries and {len(statuses)} final statuses.")

    results_csv = temp_root / args.results_file
    wrote_header = results_csv.exists()

    with results_csv.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not wrote_header:
            writer.writerow(["timestamp_filename", "prior_status", "new_status"])  # header

        for filename, ts in files.items():
            prior_status = statuses.get(filename)
            ts_filename = f"{ts}/{filename}"

            # Skip files with prior Status == no_motion
            if prior_status == "no_motion":
                logger.info(f"Skipping {ts_filename} (prior status: no_motion)")
                continue

            # Compose video path and output dir
            video_path = video_root / ts / filename
            output_dir = temp_root / ts
            output_dir.mkdir(parents=True, exist_ok=True)

            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}. Skipping.")
                writer.writerow([ts_filename, prior_status or "unknown", "file_missing"])
                csvfile.flush(); os.fsync(csvfile.fileno())
                continue

            logger.info(f"Replaying detect_motion for {ts_filename} (prior status: {prior_status or 'unknown'})")
            try:
                if args.isolate:
                    ctx = mp.get_context("spawn")
                    q = ctx.Queue()
                    p = ctx.Process(target=_worker, args=(str(video_path), str(output_dir), args.model_path or "", str(repo_root), q))
                    p.start()
                    p.join()
                    if not q.empty():
                        msg = q.get()
                    else:
                        msg = {"error": "no_result"}
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    new_status = msg.get("status", "unknown")
                else:
                    result = detect_motion_fn(str(video_path), str(output_dir))
                    new_status = result.get("status", "unknown")
                logger.info(f"Result for {ts_filename}: {new_status}")
                writer.writerow([ts_filename, prior_status or "unknown", new_status])
                csvfile.flush(); os.fsync(csvfile.fileno())
            except Exception as e:
                logger.exception(f"detect_motion failed for {ts_filename}: {e}")
                writer.writerow([ts_filename, prior_status or "unknown", "error"])
                csvfile.flush(); os.fsync(csvfile.fileno())
            finally:
                # Encourage memory reclamation in parent process between runs
                gc.collect()

    logger.info(f"Validation complete. Results written to: {results_csv}")


if __name__ == "__main__":
    main()
