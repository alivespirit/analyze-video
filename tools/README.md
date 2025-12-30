# Tools

This folder contains utility scripts to support video analysis workflows.

## Validation Script

`validate_log_replay.py` replays motion detection on videos referenced in a prior day's log using a chosen model, and records results.

### What it does
- Parses a log for entries like:
  - `[<filename>.mp4] New file detected: <timestamp>/<filename>.mp4`
  - `[<filename>.mp4] Motion detection complete. Status: <status>`
- Skips videos with prior status `no_motion`.
- Calls `detect_motion(<video_dir>/<timestamp>/<filename>, <temp_dir>/<timestamp>)`.
- Writes results to `<temp_dir>/validation_results.csv` with columns: `timestamp_filename, prior_status, new_status`.

### Memory isolation
- Use `--isolate` to run each video in a separate subprocess to fully reclaim memory after processing.
- Without `--isolate`, the script runs in-process and triggers `gc.collect()` between videos.

### Model selection
- Pass `--model-path` to test a specific detection model directory.
- The script sets `OBJECT_DETECTION_MODEL_PATH` before importing `detect_motion`, so the model is picked up at import time.

### Usage
Run from the repository root:

```bash
python tools/validate_log_replay.py \
  --log-file /path/to/previous_day.log \
  --video-dir /path/to/videos/root \
  --temp-dir /home/rmykhailiuk/analyze-video/temp \
  --model-path /path/to/new_model_dir
```

Write results to a custom filename and isolate each run:

```bash
python tools/validate_log_replay.py \
  --log-file /path/to/previous_day.log \
  --video-dir /path/to/videos/root \
  --temp-dir /home/rmykhailiuk/analyze-video/temp \
  --model-path /path/to/new_model_dir \
  --results-file validation_results_new.csv \
  --isolate
```

### Expected log format
The parser expects lines like:

```
[09M27S_1767020967.mp4] New file detected: 2025122917/09M27S_1767020967.mp4
[09M27S_1767020967.mp4] Motion detection complete. Status: no_motion
```

### Output
- Per-video results appended immediately (flush + fsync) to `<temp_dir>/<results_file>`.
- Processed videos have outputs written under `<temp_dir>/<timestamp>/`.

### Notes
- Ensure you have the repository root on `PYTHONPATH` or run from the repo root so `detect_motion` can be imported.
- Dependencies are declared in the project `requirements.txt`. Install them if running on a new environment.
