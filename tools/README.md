# Tools

This folder contains utility scripts to support video analysis workflows.

## Quick Reference

| Script | When to use | Safe/default mode | Risky mode |
|---|---|---|---|
| `run_detect_motion.py` | Test one video quickly and inspect status/clip/ReID summary | Run on a single file, outputs in `temp/` by default | None (no destructive actions) |
| `validate_log_replay.py` | Re-run a prior day log against a model and compare statuses | Append-only CSV output, optional `--isolate` per-video process | Large batch runs can be time/CPU heavy |
| `reid_gallery_dedupe.py` | Audit and clean similar ReID gallery crops | Report-only (no file changes), optional visual sheets via `--viz-dir` | `--move-duplicates-to` or `--delete-duplicates` modifies files |
| `gate_motion_detector.py` | Manual/legacy ROI motion recording utility | Local manual run after editing config constants | Hardcoded paths/settings can produce unexpected outputs if not adjusted |

Other subfolders in `tools/` contain additional project-specific utilities (`caption_watcher/`, `finetuning/`, `log_dashboard/`, `timelapse/`).

## run_detect_motion.py

Runs `detect_motion(input_video, output_dir)` for a single input video and prints a compact JSON summary to stdout.

### What it does
- Adds repo root to `sys.path` and imports `detect_motion`.
- Configures console logging for `detect_motion` logs.
- Creates output directory if missing.
- Prints summary fields such as status, clip path, counts, crossing direction, insignificant frames count, and ReID details.

### Usage

```bash
python tools/run_detect_motion.py /path/to/video.mp4
```

With custom output directory and log level:

```bash
python tools/run_detect_motion.py /path/to/video.mp4 /home/rmykhailiuk/analyze-video/temp --log-level DEBUG
```

### Arguments
- `input_video` (required): path to input `.mp4`.
- `output_dir` (optional): defaults to `<repo>/temp`.
- `--log-level`: `DEBUG|INFO|WARNING|ERROR|CRITICAL` (default: `INFO`).

### Output
- Logs to console.
- JSON summary printed to console.
- Clip and snapshots are created under `output_dir` according to `detect_motion` behavior.

## validate_log_replay.py

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

## reid_gallery_dedupe.py

Finds near-duplicate images in a ReID gallery using cosine similarity of ReID embeddings and optionally cleans duplicates.

### What it does
- Loads gallery embeddings via `PersonReID`.
- Builds similarity matrix (`cosine`) between all gallery items.
- Forms duplicate groups using threshold-connected components.
- Selects one image to keep per group (sharpest by Laplacian variance).
- Prints detailed per-group logs (`KEEP|DROP`, similarity to keep, sharpness, full path).
- Optional cleanup:
  - move duplicates to another folder
  - delete duplicates
- Optional visualization:
  - creates per-group contact sheets
  - highlights `KEEP` item with green border

### Usage

Report-only run:

```bash
python tools/reid_gallery_dedupe.py --gallery person_of_interest --similarity-threshold 0.95
```

Generate visualizations:

```bash
python tools/reid_gallery_dedupe.py \
  --gallery person_of_interest \
  --similarity-threshold 0.95 \
  --viz-dir temp/reid_viz
```

Move duplicates to a quarantine folder:

```bash
python tools/reid_gallery_dedupe.py \
  --gallery person_of_interest \
  --similarity-threshold 0.95 \
  --move-duplicates-to temp/reid_duplicates
```

### Key arguments
- `--gallery` (required): gallery directory with `.jpg/.jpeg/.png`.
- `--model`: optional path to OpenVINO ReID model XML.
- `--similarity-threshold`: pairwise cosine threshold (default `0.95`).
- `--min-group-size`: report groups of at least this size (default `2`).
- `--max-groups`: cap printed groups (`0` = all).
- `--move-duplicates-to`: move DROP files to folder.
- `--delete-duplicates`: delete DROP files.
- `--viz-dir`: output directory for group contact sheets.
- `--viz-cols`: columns in sheet (`0` = single-row).
- `--viz-tile-width`, `--viz-tile-height`: sheet tile sizes.
- `--viz-max-items`: cap rendered items per group (`0` = all).

### Notes
- `--move-duplicates-to` and `--delete-duplicates` are mutually exclusive.
- Grouping is threshold-connected (transitive), so an item can appear in a group even if its similarity to `KEEP` is below threshold.

## gate_motion_detector.py

Standalone OpenCV-based script for manual ROI drawing and basic motion-triggered clip recording.

### What it does
- Reads from a hardcoded `VIDEO_SOURCE`.
- Loads ROI polygon from `ROI_CONFIG_FILE` or asks user to draw it.
- Performs background subtraction and contour filtering.
- Records clips while motion is active and for a short tail window after motion stops.

### Important caveats
- This script currently has hardcoded Windows-style paths and constants.
- It is independent from the main `detect_motion.py` pipeline and appears to be a legacy/manual utility.
- You will likely need to edit configuration constants at the top of the script before use.

### Typical use

```bash
python tools/gate_motion_detector.py
```

### Output
- Saves motion clips into `OUTPUT_DIR` configured in the script.
- Saves ROI points JSON to `ROI_CONFIG_FILE`.
