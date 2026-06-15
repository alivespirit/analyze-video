# dataset_builder

Semi-automated YOLO dataset building: turn flagged videos (failed detections) into
**CVAT-ready, pre-annotated frames** so you only validate/fix instead of drawing from
scratch. Used to grow `/mnt/nas/training/datasets/` (classes `0=person`, `1=car`).

## Design / safety

- **Standalone.** Does *not* import `detect_motion.py` and never writes into the live
  `temp/` dirs or the processing ledger. It only *reads* the same shared assets the
  pipeline uses: the custom YOLO model and `config/roi-4k.json`. It cannot affect the
  running pipeline.
- **4K in, 4K out.** Frames are saved at native source resolution. YOLO runs on the full
  4K frame (resizing to `--imgsz` internally) and boxes are normalized, so labels are
  correct. Nothing you keep is downscaled.
- **Fast motion, the way detect_motion does it.** MOG2 on a full 4K frame is ~250 ms/frame
  (pure CPU — the GPU can't help). So motion is detected on the frame **cropped to the ROI
  and downscaled** (`--motion-max`, ~10 ms), while YOLO stays on the full frame (~15 ms on
  the GPU). Same approach the live pipeline uses.
- **Motion-in-ROI events.** Selection is restricted to frames where there's motion *inside
  the ROI* ± `--pad-seconds` (gap-merged into events) — i.e. the gate activity, not idle
  frames. `--roi none` falls back to full-frame motion.
- **Active-learning selection.** Within those event frames it does *not* dump random ones. It
  ranks each by how *hard* it is for the current model and keeps the hardest ~N, spaced out:
  - `fn_gap` — frame inside a short present→absent→present run (likely a missed detection)
  - `fn_motion` — motion in the ROI but **no** confident detection (`clean_miss` if not even an uncertain box)
  - `uncertain:N` — N boxes with confidence in `[0.10, 0.35)` (model on the fence)
  - `flicker:N` — detection count jumps vs neighbouring frames
- **Pre-annotations are a starting point, not truth.** These are the model's failure
  cases, so the auto-boxes are wrong exactly where it matters. Review every frame.

## Subcommands

```
extract        Motion+ROI gate a video/folder, pick hardest ~N frames, pre-annotate, write staging dir
pack           Turn a staging dir into a CVAT-importable YOLO 1.1 zip
list-crossings Emit gate_crossing video paths for a date range (validation-set source) via the dashboard API
```

## Full flow

### Training set (from flagged videos)

```bash
# 1) Local smoke test (slow on CPU — use a stride + --limit). Add --preview to eyeball picks.
python tools/dataset_builder/build_dataset.py extract \
    /mnt/nas/analyze-video/temp/training \
    --out /mnt/nas/training/staging/batch1 \
    --per-video 25 --scan-stride 4 --limit 2 --preview

# 2) Full run on the WORKER (Linux, GPU). Use the TensorRT engine + stride 1.
python tools/dataset_builder/build_dataset.py extract \
    /mnt/nas/analyze-video/temp/training \
    --out /mnt/nas/training/staging/batch1 \
    --model models/yolo12s.engine --per-video 25 --scan-stride 1

# 3) Package for CVAT
python tools/dataset_builder/build_dataset.py pack \
    /mnt/nas/training/staging/batch1 \
    --out /mnt/nas/training/staging/batch1_cvat.zip
```

### Validation set (last week's gate crossings)

Goal: ~10 **random** gate-crossing videos per day, a few representative frames from each.
Use `--per-day` to sample videos and **`--select even`** for frames — a validation set must be
*representative*, not biased toward the model's hard cases (`--select hard` is for training only).

```bash
# 1) Sample 10 random gate_crossing videos per day for the week (needs the dashboard up,
#    ENABLE_LOG_DASHBOARD=true, with logs retained for those days).
python tools/dataset_builder/build_dataset.py list-crossings \
    --api http://localhost:8192 --from 2026-06-07 --to 2026-06-13 \
    --per-day 10 --seed 1 --out /mnt/nas/training/staging/val_list.txt

# 2) A few representative frames from each (even spacing, not hardest), on the worker GPU.
python tools/dataset_builder/build_dataset.py extract \
    /mnt/nas/training/staging/val_list.txt \
    --out /mnt/nas/training/staging/val_batch \
    --model models/yolo12s.engine --select even --per-video 5 --scan-stride 1

# 3) Package for CVAT.
python tools/dataset_builder/build_dataset.py pack /mnt/nas/training/staging/val_batch
```

10 videos/day × 7 days × 5 frames ≈ **350 val frames** to validate/fix — adjust `--per-day`
/`--per-video` to taste. Re-run with the same `--seed` for a reproducible sample.

## Import into cvat.ai

The zip is a **YOLO 1.1** dataset (images + labels). Two ways:

- **A.** Create a task by uploading the `images/` folder (your usual step), then
  *Actions → Upload annotations → format `YOLO 1.1` →* the zip.
- **B.** *Create from dataset → format `YOLO 1.1` →* the zip (images + pre-annotations in one shot).

Then validate/fix boxes and export. The CVAT project's labels **must** be `person`, `car`
in that order so class ids line up with the existing dataset.

## Merge into the dataset & retrain

After exporting the fixed annotations from CVAT (YOLO format), copy images→
`/mnt/nas/training/datasets/images/{train,val}/` and labels→ `.../labels/{train,val}/`
(frame names are `{video_stem}_frame_{idx}` so they merge without collisions), then run
training per `tools/finetuning/README.md`.

> ⚠️ **No train/val leakage.** Keep frames from any one video entirely in train *or* val,
> and don't reuse a video that's already in the training batch as a val source.

## Useful knobs (see `--help`)

| Flag | Default | Note |
|------|---------|------|
| `--select` | hard | `hard` = active-learning hardest frames (**training**); `even`/`random` = representative content frames (**validation**) |
| `--per-video` | 25 | target frames/video (use a few, e.g. 4–5, for validation) |
| `--roi` | config/roi-4k.json | motion is detected WITHIN this polygon (cropped+downscaled for speed); `none` = full-frame motion |
| `--pad-seconds` | 3.0 | keep frames within N s before/after ROI motion (the event window) |
| `--max-gap-seconds` | 3.0 | merge ROI-motion bursts closer than this into one event |
| `--motion-max` | 960 | longest side of the throwaway frame MOG2 runs on (the main speed knob); does not affect YOLO or saved images |
| `--work-max` | 0 (full) | longest side for the YOLO frame; detection-neutral (it resizes to `--imgsz`); saved images always full-res |
| `--imgsz` | 640 | YOLO inference size = detection detail; raise (+ re-export engine) for small/distant objects |
| `--scan-stride` | 2 | scan every Nth frame; raise to go faster, 1 on the GPU worker |
| `--min-spacing` | 8 | min frame gap between picks — the main diversity guarantee |
| `--yolo-on-motion` | off | skip YOLO on frames with no ROI motion (a bit faster) |
| `--copy-local` | off | copy each clip to /tmp once before the decode passes (fewer NAS reads for big batches) |
| `--dhash-thresh` | -1 (off) | optional visual near-duplicate dedup; a whole-frame hash misses small/distant movers, so leave off unless a scene has long identical stretches |
| `--topup motion` | off | if too few hard frames, top up with high-motion frames |
| `--export-conf` | 0.35 | confidence floor for written pre-annotation boxes |
| `--dry-run` | off | score+select only (writes manifest, no images) — fast for tuning |
| `--preview` | off | also write downscaled annotated previews to `<out>/preview/` |

Each run appends a `manifest.jsonl` (video, frame, score, reasons, boxes exported) to the
staging dir for traceability.
