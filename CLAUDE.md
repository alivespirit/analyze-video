# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Surveillance video analysis system ("Споглядайко") that monitors a folder for new `.mp4` files, detects motion and objects, generates AI descriptions via Google Gemini, and sends results to Telegram. Designed for a residential gate/entrance camera with gate crossing detection, person re-identification, and optional Tesla integration.

Companion Android dashboard app: [analyze-video-dashboard](https://github.com/alivespirit/analyze-video-dashboard) (Jetpack Compose, consumes the JSON API from `tools/log_dashboard/`).

## Running the Application

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run (requires .env with GEMINI_API_KEY, TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, VIDEO_FOLDER)
python main.py
```

The app auto-restarts when any `.py` file is modified (via watchdog + `os.execv`).

## Testing a Single Video

```bash
python tools/run_detect_motion.py /path/to/video.mp4 [output_dir] [--log-level DEBUG]
```

There is no automated test suite. Validation is done via `tools/validate_log_replay.py` for batch re-analysis.

## Architecture

**Pipeline**: File monitoring → Motion detection (local or remote worker) → AI analysis → Telegram notification

**Executor lanes** (critical design choice):
- `motion_executor`: ThreadPoolExecutor(max_workers=1) — CPU-bound local motion detection runs serially
- The analysis stage runs in one of **three status-routed lanes** (separate pools so they can't head-of-line block each other; each pool size is the rate-limiter for its resource):
  - `fast_executor` (`FAST_ANALYSIS_WORKERS`, default 4) — instant work: `no_motion`, `gate_crossing` (pure formatting, no external call)
  - `llm_executor` (`LOCAL_LLM_MAX_WORKERS`, default 1) — `no_person`, `no_significant_motion` (local Ollama vision LLM; the worker's single GPU serializes, so 1)
  - `gemini_executor` (`GEMINI_MAX_WORKERS`, default 1) — `significant_motion`, `error` (Gemini API; small to stay under RPM quota; off-peak these return instant placeholders)
  - Routing is by `motion_result['status']` in `main.py`'s process loop. Net effect: a `gate_crossing` returns instantly even while several `no_person` clips sit in the local LLM, and Gemini runs **in parallel** with the local LLM instead of behind it.
- Remote worker dispatch is fully async (no executor slot consumed while waiting on HTTP)
- All executors are driven by a single asyncio event loop in `main.py`

### Core Modules

| Module | Role |
|--------|------|
| `main.py` | Entry point, asyncio orchestration, watchdog file monitoring, Telegram bot setup, processing ledger, auto-restart, retention cleanup, Tesla SoC scheduler |
| `detect_motion.py` | Core vision: background subtraction, YOLO tracking (OpenVINO), gate crossing detection, ReID triggering, highlight clip generation, car speed estimation |
| `analyze_video.py` | Gemini API calls with dynamic model selection (time-based Pro vs Flash), fallback chains, prompt loading from `config/prompt.txt`. Routes `no_person`/`no_significant_motion` videos to the local vision model (`analyze_frame.py`) instead of Gemini |
| `analyze_frame.py` | Local frame description via Ollama (qwen3-vl on the worker GPU). Analyzes saved event frames for low-motion videos; English output from `config/prompt_frame.txt`. Falls back to a placeholder when Ollama is unavailable |
| `telegram_notification.py` | Message delivery with retry logic, grouped notifications, inline callbacks for full video, media validation. Sends `no_person`/`no_significant_motion` as a single event-frame photo (downscaled) + description caption + "Глянути" button (gated by `SEND_INSIGNIFICANT_FRAMES`); falls back to the grouped text message when off. Respects `KEEP_HIGHLIGHTS_CLIPS` for clip cleanup. |
| `person_id.py` | Intel OpenVINO ReID model, gallery embedding with disk caching (keyed by gallery path + model path), negative gallery support, cosine similarity matching |
| `path_utils.py` | Timestamp extraction from video filenames |
| `worker/client.py` | Master-side worker dispatch: async HTTP, health check with caching, local fallback (lazy YOLO load), log replay with `[W]` tagging, Wake-on-LAN |
| `worker/server.py` | FastAPI worker server: path translation (Windows↔Linux), video copy, detect_motion, result copy back to CIFS, gallery cache pre-warming |
| `tools/log_dashboard/app.py` | FastAPI dashboard: HTML log viewer + JSON API for Android app (per-video summaries, stats, monitoring, events, image/clip serving, ReID gallery management) |

### Data Flow

1. `main.py` watchdog detects new `.mp4` → checks worker availability
2. If worker available: async HTTP dispatch to `worker/server.py`; otherwise queues to `motion_executor` locally
3. `detect_motion.py` returns result dict with status, clip path, person/car counts, crossing info, ReID results
4. `analyze_video.py` decides whether to call Gemini based on motion status and time of day
5. `telegram_notification.py` sends animation/photo/text with retry and grouping logic

### Directory Structure

```
analyze-video/
├── main.py, detect_motion.py, analyze_video.py   # core pipeline
├── telegram_notification.py, person_id.py        # core pipeline
├── path_utils.py                                 # utility
├── config/                                       # runtime configuration files
│   ├── roi.json, roi-1080p.json, roi-4k.json     # ROI polygons
│   ├── tracker.yaml                              # ByteTrack/BoTSORT config
│   ├── gemini_models.env                         # Gemini model names (hot-reload)
│   └── prompt.txt                               # Gemini prompt (Ukrainian)
├── worker/                                       # remote worker package
│   ├── server.py                                 # FastAPI worker server
│   ├── client.py                                 # master-side dispatch client + WOL
│   ├── worker.service                            # systemd unit
│   └── requirements.txt                          # worker-only dependencies
├── tools/                                        # dev/ops utilities
│   └── log_dashboard/                            # HTML dashboard + JSON API
│       ├── app.py                                # FastAPI app (HTML routes + /api/* JSON routes)
│       ├── templates/                            # Jinja2 HTML templates
│       └── static/                               # CSS
├── models/                                       # YOLO + ReID model files
└── temp/                                         # runtime temp files, caches, and daily dirs
    ├── processing_ledger.json                    # file processing status tracker
    ├── tesla_soc.txt                             # Tesla SoC cache
    └── YYYYMMDD/                                 # daily dirs with highlight clips, frames, ReID crops
```

### Key Configuration

- **`.env`**: All secrets and runtime config (API keys, paths, thresholds)
- **`config/gemini_models.env`**: Gemini model names, re-read on every call (no restart needed)
- **`config/roi-1080p.json` / `config/roi-4k.json`**: Resolution-specific ROI polygons for motion detection and tracking
- **`config/tracker.yaml`**: ByteTrack/BoTSORT tracking parameters
- **`config/prompt.txt`**: Gemini prompt (in Ukrainian)

### Resolution Handling

`detect_motion.py` has a `RES_CONFIGS` dict with per-resolution settings (1080p, 4K). ROI configs are loaded from resolution-specific JSON files in `config/` with fallback to `config/roi.json`. 4K frames are downscaled to 1080p for highlight output. Insignificant/no_person frames are saved at original resolution (4K).

### Remote Worker

Motion detection can be offloaded to a worker machine over HTTP (`worker/server.py`, FastAPI). The master dispatches asynchronously via `worker/client.py` and falls back to local processing if the worker is unavailable. Worker logs are replayed into the master log with `[W]` inserted after the `[filename]` bracket. See `worker/README.md` for setup.

Key worker env vars: `WORKER_ENABLED`, `WORKER_URL`, `WORKER_TIMEOUT`, `WORKER_MIN_BATTERY`.

The worker `/health` endpoint reports: status, active/max tasks, battery percent, load averages (1m/5m/15m), memory usage, and CPU temperature (Package id 0 from coretemp).

### Wake-on-LAN

The master can automatically wake the worker when power is restored. When the worker health check fails and the master is plugged in (AC power), a WOL magic packet is sent via the configured network interface with a 5-minute cooldown between attempts. Implemented in `worker/client.py`.

Key env vars: `WORKER_WAKE_ON_LAN` (bool, default false), `WORKER_WAKE_ON_LAN_MAC` (MAC address), `WORKER_WAKE_ON_LAN_IFACE_IP` (bind IP for WOL broadcast, default `10.0.0.1`).

### Highlight Clips and Daily Directories

Highlight clips are saved to daily subdirectories under `TEMP_DIR` (`temp/YYYYMMDD/<video>.mp4`). The `KEEP_HIGHLIGHTS_CLIPS` env var (default `true`) controls whether clips are preserved after Telegram send — when true, clips remain for viewing in the Android dashboard and are cleaned up by the daily directory retention process. Both `main.py` and `telegram_notification.py` check this setting.

Insignificant/no_person frames and ReID crops are also saved to these daily directories:
- Frames: `{hour}H{video_stem}_{tag}_{frameIdx}.jpg` (tag = `insignificant` or `no_person`)
- ReID crops: `{video_stem}_reid_best{N}.jpg`

### Gate-Crossing Crop Magnifier (highlight PiP)

A live picture-in-picture magnifier of the person at the gate, baked into the highlight clip pixels entirely within the existing per-frame draw loop in `detect_motion.py` — no buffering, no second pass, no dependency on the post-hoc ReID best-crop selection (which runs once at the end of the video, *after* frames are already written, so it isn't available at write time).

Each appended frame: the largest person whose bbox center is within the gate band is cropped from a **clean (un-annotated) frame copy** (`clean_frame_overlay`, reusing the existing `frame_for_reid` copy when present), zoomed (aspect-preserving) into the lower-right corner **above** the event text — `draw_event_overlay` returns its box top-y so the PiP anchors above it — with a two-line "funnel" connector (box top-right→PiP top-left, box bottom-right→PiP bottom-left). The PiP border and funnel lines use the **same color as the drawn bounding box**: `COLOR_PERSON` (green) normally, `COLOR_HIGHLIGHT` (red) when the person is in `LINE_Y_TOLERANCE` or within the `HIGHLIGHT_WINDOW_FRAMES` highlight window. The connector lines are drawn before the crop/background so they clip cleanly at the PiP edge. Helper: `draw_crop_overlay()`.

Visibility is positional, not crossing-confirmed: the PiP shows while the person's center is within the gate band (so it appears as they approach/cross and disappears once they move away), not while they linger far from the gate. The band defaults to `(LINE_Y_TOLERANCE + REID_LINE_EXTRA_TOLERANCE) * GATE_CROP_OVERLAY_BAND_SCALE` (resolution-aware); the bare tolerance is only a few frames wide, so the scale (default `4.0`) is the knob that keeps the PiP on screen long enough.

Drawn on the orig-resolution frame before the existing resize-to-1080p + append path, so 4K and 1080p both carry through unchanged.

Key env vars (master, read at module load): `GATE_CROP_OVERLAY_ENABLED` (default `true`), `GATE_CROP_OVERLAY_WIDTH_FRAC` (default `0.18`, PiP width as a fraction of frame width), `GATE_CROP_OVERLAY_BAND` (absolute px override; `-1` = derive at runtime), `GATE_CROP_OVERLAY_BAND_SCALE` (default `4.0`, multiplier on the derived band — bigger = visible longer), `GATE_CROP_OVERLAY_UPPER_BODY` (default `false`, crop only the top 50% / head+torso). The connector geometry uses the full bbox even when `UPPER_BODY` crops only the top half.

### Tesla SoC

Tesla State of Charge is fetched by a periodic scheduler in `main.py` (`tesla_soc_scheduler`) and written to a cache file (`TESLA_SOC_FILE`, default `temp/tesla_soc.txt`). `detect_motion.py` only reads the cache — it has no Tesla credentials or API calls.

### ReID Gallery Cache

`person_id.py` caches gallery embeddings as NPZ files in `REID_CACHE_DIR` (default `temp/`). The cache filename includes a hash of both the gallery path and the model path, so master and worker (which use different ReID models) maintain separate cache files even when `REID_CACHE_DIR` points to shared NAS storage. The worker pre-warms its cache at startup and periodically via a background loop in `worker/server.py` (`GALLERY_PRECHECK_INTERVAL`, default 60 s).

### Restart Recovery

A JSON ledger (`temp/processing_ledger.json`) tracks file processing status. After restart, files within `RESTART_RECOVERY_WINDOW_SECONDS` (default 180s) are re-queued.

### Fast Processing Mode

When motion queue backlog exceeds a threshold, adaptive frame skipping kicks in — controlled by `FAST_MOTION_STRIDE`, `FAST_TRACK_FULL_UNTIL_SECONDS`, `FAST_TRACK_SKIP_FROM_SECONDS` env vars.

### Local Frame Analysis (Ollama / qwen3-vl)

Wind/sun produce many low-motion recordings (`no_person`, `no_significant_motion`) that used to consume Gemini's daily quota. These are now described by a local vision model (Ollama) running on the worker GPU, freeing Gemini for `significant_motion` videos. `analyze_frame.py` reads the saved event frames, base64-encodes them, and POSTs to Ollama's `/api/generate`. Runs at all hours (no quota); falls back to the original placeholder messages if Ollama is unreachable.

**Multi-frame mode (default).** When a video has ≥2 event frames they go to Ollama in ONE multi-image call (multi-frame prompt), so the model sees the whole short clip and can narrate progression and dedup naturally — more coherent than per-frame, and for reasoning models much faster (one reasoning pass instead of N + a merge). A 1-frame video is a single-image call. Set `OLLAMA_MULTI_FRAME=false` to fall back to the legacy per-frame analysis + text-only combine merge.

`detect_motion` always collects saved event-frame paths into the result dict field `event_frames` (gated only by `SAVE_INSIGNIFICANT_FRAMES`, i.e. whether frames are saved to disk at all). `analyze_video` picks the most detailed one via `_pick_best_frame` and returns it as `photo_frame`; the worker translates `event_frames` to master-perspective in `translate_result_paths`. (The old always-empty `insignificant_frames` list/parameter has been removed end-to-end.)

**Telegram delivery of frame descriptions.** When `analyze_frames_local` returns a description, the result carries `photo_frame` (the chosen frame). `telegram_notification.send_notifications` then sends a single photo (frame) with the description as caption and a "Глянути" full-video button — a standalone photo, unlike a media-group album, *can* carry an inline keyboard. Gated by `SEND_INSIGNIFICANT_FRAMES` (default `true`, read in `telegram_notification.py`); when `false`, the description falls through to the grouped text message. The frame is downscaled before upload via `TELEGRAM_FRAME_MAX_DIM` (default `1920`, longest side; `0` disables) / `TELEGRAM_FRAME_JPEG_QUALITY` (default `88`) — the on-disk frame stays full-res for the dashboard. The caption is clamped to Telegram's 1024-char limit (`_truncate_caption`), so keep the prompt's output under ~900 chars. If the photo send fails it falls back to a plain text message with the button (on the final retry), mirroring the animation path.

Key env vars (master): `LOCAL_FRAME_ANALYSIS_ENABLED` (default `true`), `OLLAMA_URL` (default `http://10.0.0.2:11434`), `OLLAMA_MODEL` (default `qwen3-vl:2b-instruct` — use the `-instruct` variant; the plain `qwen3-vl:2b` has mandatory chain-of-thought that breaks `num_predict` and adds 30-60s of latency), `OLLAMA_TIMEOUT` (default `60`), `OLLAMA_ATTEMPTS` (default `2`, total tries per frame — retries on a request error or empty response), `OLLAMA_IMAGE_MAX_DIM` (default `1280`, longest side; frames are downscaled before sending — a 4K frame's huge vision-patch count costs the model ~30s to encode, vs ~11s at 1280 / ~6s at 896, with grounding intact; the on-disk frame is untouched; set `0` to disable). Sampling knobs: `OLLAMA_TEMPERATURE` (default `0.8`, high for livelier/funnier output), `OLLAMA_TOP_K` (default `20`) and `OLLAMA_TOP_P` (default `0.92`) — the tight top_k/top_p clamp is what keeps the high temperature coherent (blocks the long-tail tokens that otherwise cause the 2B model to ramble/loop), `OLLAMA_NUM_PREDICT` (default `80`, caps output length — prevents the small model from rambling until it fills the context and gets aborted), `OLLAMA_REPEAT_PENALTY` (default `1.3`), `OLLAMA_MAX_CHARS` (default `300`) — a response longer than this is treated as a rambling/repetition loop (the high-temperature failure mode) and triggers a retry via `OLLAMA_ATTEMPTS`, `OLLAMA_MULTI_FRAME` (default `true`, see multi-frame mode above; `false` = legacy per-frame + combine), `OLLAMA_MULTI_MAX_FRAMES` (default `4`, cap on images per multi-frame call — extras are sampled evenly across the clip and the drop is logged), `OLLAMA_COMBINE_FRAMES` (default `true`; only used in legacy mode — merges per-frame sentences via a text-only follow-up call), `OLLAMA_THINK` (default unset/omitted; set `true` for reasoning models like gemma4 — with `OLLAMA_NUM_PREDICT=-1` — see below), `OLLAMA_KEEP_ALIVE` (default `-1` = model resident forever; set e.g. `30m` to unload during idle hours and free RAM/GPU — accepts int seconds or a duration string), `OLLAMA_PROMPT_FILE` / `OLLAMA_MULTI_PROMPT_FILE` / `OLLAMA_COMBINE_PROMPT_FILE` (override prompt file paths to swap prompts per model without editing the defaults). Worker setup: run `ollama serve` with `OLLAMA_HOST=0.0.0.0:11434`; the client sends `keep_alive` per request (from `OLLAMA_KEEP_ALIVE`), which overrides the server's own keep-alive. Prompts (all English defaults, re-read every call so edits take effect without restart): `config/prompt_frame.txt` (single frame), `config/prompt_frame_multi.txt` (multi-frame), `config/prompt_frame_combine.txt` (legacy merge).

#### Alternative model: gemma4 (Ukrainian output)

`gemma4:e4b-it-q4_K_M` produces good Ukrainian (where qwen3-vl:2b is weak), at the cost of speed. Two model-specific gotchas, both handled by env only (no code change):

- **Reasoning is the quality/speed dial.** Gemma 4 is a reasoning model: on image calls it emits ~700-1000 hidden reasoning tokens (Ollama 0.20.3 discards them — not in `response` or `thinking`) before the answer. `OLLAMA_THINK=true` (with `OLLAMA_NUM_PREDICT=-1` so the reasoning isn't truncated to an empty response) gives well-grounded Ukrainian — it reads timestamps off the frame and rarely fabricates — but costs ~50-75s/frame (CPU-bound; see below). `OLLAMA_THINK=false` is ~5-10x faster (`eval_count` ~800→~40, normal `num_predict` works) but hallucinates absent objects and produces shakier grammar. We chose thinking ON: these are low-priority clips with no quota, so quality wins over latency. Note the throughput risk: at ~2-4 min/video, busy windy days can back the queue up.
- **It does not fit 4 GB VRAM.** E4B is ~8B total params (Per-Layer Embeddings); the q4 build loads to ~10.6 GB with only ~3 GB in VRAM, so it runs mostly on CPU. Higher precision (q8 = 12 GB) is slower, not better; the base (non-`it`) variant follows the prompt worse. E4B-q4-it is the sweet spot for quality on this hardware; the smaller E2B fits VRAM better (faster) at lower quality.

Multi-frame mode helps gemma especially: one reasoning pass over all frames instead of N, so a 3-frame video drops from ~4 min to ~1.5 min, and the model narrates the clip's progression coherently (validated to be both faster and better-grounded than per-frame+combine).

Recommended gemma env block: `OLLAMA_MODEL=gemma4:e4b-it-q4_K_M`, `OLLAMA_THINK=true`, `OLLAMA_NUM_PREDICT=-1`, `OLLAMA_PROMPT_FILE=config/prompt_frame_uk.txt`, `OLLAMA_MULTI_PROMPT_FILE=config/prompt_frame_multi_uk.txt`, `OLLAMA_MAX_CHARS=450` (Ukrainian runs longer), `OLLAMA_TIMEOUT=200` (thinking ON can take ~75s/frame), and optionally `OLLAMA_KEEP_ALIVE=30m` to free RAM/GPU during idle hours. Ukrainian prompts live in `config/prompt_frame_uk.txt`, `config/prompt_frame_multi_uk.txt`, `config/prompt_frame_combine_uk.txt`.

#### Two-stage mode: fast vision + Ukrainian refiner (recommended for clean Ukrainian)

Setting `OLLAMA_REFINE_MODEL` switches `analyze_frame.py` to a two-stage pipeline that gives the cleanest, russism-free Ukrainian:
1. **Stage 1 — vision (English):** `OLLAMA_MODEL` (a fast vision model, e.g. `qwen3-vl:2b-instruct`) describes each frame **per-frame, single-image** (the 2B model mishandles multi-image input, so multi-frame mode is bypassed in this path). English; grounding only — language doesn't matter here.
2. **Stage 2 — refine (Ukrainian):** `OLLAMA_REFINE_MODEL` (a dedicated Ukrainian model, e.g. `lapa` — `hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M`) merges the per-frame English descriptions into ONE cheeky Ukrainian sentence via a text-only call. lapa is Gemma-3-12B-based and native-Ukrainian, so it writes idiomatic, russism-free output and even cleans russisms in its input. The refine prompt (`config/prompt_frame_refine_uk.txt`) also strips any "camera" mentions the vision model leaks.

Why two models: lapa is a strong *text* Ukrainian model but its GGUF release ships no vision projector (`mmproj`), so it can't see images — it does the writing, a small vision model does the seeing. **VRAM note:** lapa (~9 GB) and qwen can't both sit in the 4 GB GPU; lapa's mere residency slows qwen even when qwen is 100% GPU. Set `OLLAMA_REFINE_NUM_GPU=0` to run lapa fully on CPU (frees GPU for qwen). Expect ~30s/frame + ~15-37s refine ≈ ~45s (1 frame) to ~130s (multi-frame) — quality-over-speed, as chosen.

Two-stage env vars: `OLLAMA_REFINE_MODEL` (unset = single-model mode; set = enable two-stage), `OLLAMA_REFINE_PROMPT_FILE` (default `config/prompt_frame_refine_uk.txt`; uses a `{descriptions}` placeholder), `OLLAMA_REFINE_NUM_GPU` (set `0` to force the refiner onto CPU), `OLLAMA_REFINE_KEEP_ALIVE` (overrides keep-alive for the refine model only; set `-1` so lapa — CPU-resident, 0 VRAM — never unloads and never pays its ~50s cold load; ~free given the worker's RAM headroom), `OLLAMA_REFINE_TEMPERATURE` (default `0.8`), `OLLAMA_REFINE_NUM_PREDICT` (default `120`), `OLLAMA_REFINE_REPEAT_PENALTY` (default `1.1`). Stage 1 uses the standard `OLLAMA_*` knobs; `OLLAMA_MAX_CHARS` (raise to ~450 for Ukrainian) guards both stages.

**Switching between the two setups (env-only, no code change):**
- *Two-stage (clean Ukrainian, recommended):* `OLLAMA_MODEL=qwen3-vl:2b-instruct`, `OLLAMA_PROMPT_FILE=config/prompt_frame.txt` (English), `OLLAMA_REFINE_MODEL=hf.co/lapa-llm/lapa-v0.1.2-instruct-GGUF:Q4_K_M`, `OLLAMA_REFINE_NUM_GPU=0`, `OLLAMA_REFINE_KEEP_ALIVE=-1` (keep lapa resident — kills its ~50s cold load; free on a 32 GB worker since it's 0 VRAM), `OLLAMA_MAX_CHARS=450`, `OLLAMA_TIMEOUT=200`. (Leave `OLLAMA_THINK` unset — qwen doesn't reason.)
- *Single-model gemma4:* **unset `OLLAMA_REFINE_MODEL`** and use the gemma env block above.

## Log Dashboard

The log dashboard (`tools/log_dashboard/app.py`) serves two roles:
1. **HTML dashboard** — browser-based log viewer with per-day insights, filters, charts, and inline video player
2. **JSON API** — consumed by the Android dashboard app

Enable via `ENABLE_LOG_DASHBOARD=true`. Default port: `8192`.

### JSON API endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/days` | List of available log days (YYYY-MM-DD) |
| `GET /api/today/videos?day=` | Per-video summary (status, gate, ReID, processing time, `has_frames` indicator) |
| `GET /api/today/video/{basename}/logs?day=` | Log entries for a specific video |
| `GET /api/today/video/{basename}/reid-crops` | ReID crop image URLs (from TEMP_DIR daily dirs) |
| `GET /api/today/video/{basename}/frames` | Insignificant/no_person frame URLs (from TEMP_DIR daily dirs) |
| `GET /api/today/video/{basename}/highlight` | Highlight clip URL if available (from TEMP_DIR daily dirs) |
| `GET /api/today/gate-crossings?day=` | Videos with ReID crops: basename, time, direction, status, scores, crop URLs |
| `GET /api/today/stats?day=` | Aggregated stats (status counts, gate counts, processing times, away/back intervals). For today, open away intervals get `dur` filled with elapsed time and an `ongoing: true` flag |
| `GET /api/stats/overall` | Overall stats with per-day data, events heatmap, weekday heatmap. Heatmaps use the union of (log-file days, cached days) excluding today so the pattern window survives log retention; weekday cells include per-bin `away_occurrences` / `back_occurrences` |
| `GET /api/stats/reid` | ReID auto-detection accuracy per day: TP/FP/FN, precision/recall/F1, average match score, 7-day MA. Each per-day entry also returns `events: [{video, hhmmss, kind: TP\|FP\|FN\|FPFN, score, crop_url}]` for the recognition-wall view. Persisted in `temp/reid_metrics_cache.json` (v3) and never pruned |
| `GET /api/monitoring` | Master CPU/RAM/battery + worker health proxy + recent processing ledger |
| `GET /api/events/latest?since=` | Away/back events with current home/away status (for notifications). May include `next_prediction: {kind: "away"\|"back", predicted_hhmm, confidence, basis_count, basis_total, imminent}` when the same-weekday pattern has ≥4 samples, ≥30% confidence, and the predicted time is within a 3-hour horizon (back predictions also require ≥20 min after the away start) |
| `POST /api/reid/copy` | Copy ReID crop to positive or negative gallery |
| `GET /api/image/{basename}` | Serve images (crops, frames) from TEMP_DIR or VIDEO_FOLDER |
| `GET /api/highlight/{basename}` | Serve highlight clips from TEMP_DIR daily dirs |

All `?day=` parameters default to today (or latest available day). API responses use per-day caching (30s TTL, max 7 days in memory). The `_walk_daily_dirs()` helper restricts file searches to `YYYYMMDD`-named subdirectories of `TEMP_DIR` to avoid scanning unrelated folders (e.g., `training/`).

The `has_frames` field in `/api/today/videos` is derived from "Saved ... frame to" log messages via `collect_metrics()`, avoiding filesystem scans during video list construction.

Worker health is proxied via `_get_worker_health()` with 30s cache. When the worker is configured (`WORKER_URL` set) but unreachable, returns `{"status": "offline"}` instead of `null`.

`/api/monitoring` reads master stats from `psutil` — `cpu_percent(interval=None)` (non-blocking, 10s cache), `virtual_memory()`, `sensors_battery()`.

## Language Note

User-facing messages, Telegram captions, and the Gemini prompt are in Ukrainian.

## Tools Directory

- `tools/run_detect_motion.py` — Single video analysis CLI
- `tools/validate_log_replay.py` — Batch re-analysis from logs
- `tools/reid_gallery_dedupe.py` — Find duplicate ReID gallery images
- `tools/finetuning/` — YOLO training and export scripts
- `tools/log_dashboard/` — FastAPI web dashboard + JSON API for Android app (enable via `ENABLE_LOG_DASHBOARD=true`)
