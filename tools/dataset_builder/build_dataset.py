#!/usr/bin/env python3
"""
dataset_builder — semi-automated YOLO dataset frame extraction + pre-annotation.

Purpose
-------
Turn flagged videos (failed detections) into CVAT-ready, pre-annotated frames so
the human only validates/fixes instead of drawing from scratch.

Design constraints (deliberate)
-------------------------------
* STANDALONE. It does NOT import detect_motion.py and never writes into the live
  temp/ dirs or the processing ledger. It only *reads* the same shared assets the
  pipeline uses: the custom YOLO model and config/roi-4k.json. So it cannot affect
  the running pipeline.
* 4K in, 4K out. Frames are saved at native source resolution; YOLO returns boxes
  in native pixels (it letterboxes internally for imgsz), so normalized labels are
  correct regardless of imgsz.
* Active-learning frame selection. It does NOT just dump motion frames — it ranks
  frames by how hard they are for the current model (false-negative gaps, motion
  with no detection, low-confidence/uncertain boxes, detection-count flicker) and
  keeps the hardest ~N, spaced out and de-duplicated. Those are the frames worth a
  human's labeling time.

Subcommands
-----------
  extract        Motion+ROI gate a video (or folder of videos), pick the hardest
                 ~N frames, pre-annotate with YOLO, write a staging dir.
  pack           Turn a staging dir into a CVAT-importable YOLO 1.1 dataset zip.
  list-crossings Emit gate_crossing video paths for a date range (validation-set
                 source), read from the dashboard JSON API.

Typical flow
------------
  # 1) local smoke test (slow on CPU; use a stride and a couple of videos)
  python tools/dataset_builder/build_dataset.py extract \
      /mnt/nas/analyze-video/temp/training \
      --out /mnt/nas/training/staging/batch1 --per-video 25 --scan-stride 3 --limit 2

  # 2) full run on the worker (GPU engine, stride 1)
  python tools/dataset_builder/build_dataset.py extract \
      /mnt/nas/analyze-video/temp/training \
      --out /mnt/nas/training/staging/batch1 \
      --model models/yolo12s.engine --per-video 25 --scan-stride 1

  # 3) package for CVAT
  python tools/dataset_builder/build_dataset.py pack \
      /mnt/nas/training/staging/batch1 --out /mnt/nas/training/staging/batch1_cvat.zip
"""
import os
import sys
import json
import glob
import math
import time
import zipfile
import logging
import argparse
import urllib.request
from datetime import datetime, date, timedelta

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # tools/dataset_builder -> tools -> repo

logger = logging.getLogger("dataset_builder")

# --- Defaults mirroring the main pipeline (config files / model are the real reuse) ---
DEFAULT_MODEL = os.path.join("models", "yolo12n_openvino_model")  # OpenVINO (CPU) here; pass .engine on the worker
DEFAULT_ROI = os.path.join("config", "roi-4k.json")  # motion is detected WITHIN this ROI (the gate band),
#                       cropped+downscaled like detect_motion for speed. Pass --roi none for full-frame motion.
CLASS_NAMES = ["person", "car"]            # index order MUST match the dataset (0=person, 1=car)
DETECT_CLASSES = [0, 1]
DEFAULT_IMGSZ = 640

# --- Active-learning scoring knobs (tunable; sensible defaults) ---
SCAN_CONF = 0.10        # predict floor — low, so we can SEE uncertain/near-miss boxes
PRESENT_CONF = 0.35     # a box this confident counts as a "real" detection (== pipeline CONF_THRESHOLD)
MOTION_FRAC_THRESH = 0.003   # fraction of ROI pixels in motion to call a frame "active"
W_FN_GAP = 5.0          # frame inside a short present->absent->present gap (likely missed detection)
W_FN_MOTION = 4.0       # motion in ROI but zero confident detections
W_FN_MOTION_EMPTY = 2.0 # extra: not even an uncertain box (a clean miss)
W_UNCERTAIN = 2.0       # per uncertain box (conf in [SCAN_CONF, PRESENT_CONF))
W_FLICKER = 1.5         # per unit of detection-count change vs neighbours
W_MOTION_TIE = 0.5      # small tiebreak so busier frames win ties


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def resolve_path(p: str) -> str:
    """Allow paths relative to the repo root so `models/...` works from anywhere."""
    if os.path.isabs(p) or os.path.exists(p):
        return p
    cand = os.path.join(REPO_ROOT, p)
    return cand if os.path.exists(cand) else p


def load_roi_polygon(roi_path: str):
    """Read config/roi-*.json and return the motion_detection_roi polygon as int32 Nx2.

    Inlined (not imported from detect_motion) to keep this tool fully decoupled.
    Supports the dict form ({'motion_detection_roi': [...]}) and the legacy list form.
    """
    with open(roi_path, "r") as f:
        cfg = json.load(f)
    if isinstance(cfg, dict):
        pts = cfg.get("motion_detection_roi")
    elif isinstance(cfg, list):
        pts = cfg
    else:
        pts = None
    if not pts:
        raise ValueError(f"No motion_detection_roi polygon found in {roi_path}")
    return np.array(pts, dtype=np.int32)


def fit_polygon_to_frame(pts: np.ndarray, w: int, h: int) -> np.ndarray:
    """If the ROI (authored for 4K) overshoots a smaller frame, downscale it to fit.

    For native 4K input against a 4K ROI this is a no-op.
    """
    if pts is None or len(pts) == 0:
        return pts
    max_x, max_y = int(pts[:, 0].max()), int(pts[:, 1].max())
    if max_x <= w and max_y <= h:
        return pts
    looks_4k = max_x > 2500 or max_y > 1400
    base_w, base_h = (3840, 2160) if looks_4k else (1920, 1080)
    scaled = pts.astype(np.float32)
    scaled[:, 0] *= w / base_w
    scaled[:, 1] *= h / base_h
    scaled[:, 0] = np.clip(scaled[:, 0], 0, w - 1)
    scaled[:, 1] = np.clip(scaled[:, 1], 0, h - 1)
    logger.warning("ROI scaled to fit %dx%d frame (authored for %dx%d)", w, h, base_w, base_h)
    return scaled.astype(np.int32)


def dhash(frame, hash_size: int = 8) -> int:
    """64-bit difference hash for near-duplicate detection (resolution-independent)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def load_model(model_path: str):
    from ultralytics import YOLO  # imported lazily so `pack`/`list-crossings` don't need it
    mp = resolve_path(model_path)
    logger.info("Loading YOLO model: %s", mp)
    return YOLO(mp, task="detect")


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #
def _scaled_dims(W, H, longest_max):
    """Return (w, h) scaled so the longest side <= longest_max; full size if longest_max falsy."""
    longest = max(W, H)
    if longest_max and longest > longest_max:
        sc = longest_max / float(longest)
        return max(1, int(round(W * sc))), max(1, int(round(H * sc)))
    return W, H


def scan_video(video_path, model, roi_pts, args):
    """First pass: fast ROI-cropped motion gate (mirrors detect_motion) + YOLO; store records.

    Motion detection is made cheap exactly the way detect_motion does it: the frame is CROPPED to
    the ROI bounding box (+ a little padding) and downscaled to `--motion-max` before background
    subtraction, so MOG2 runs on a tiny image instead of the full 4K frame (~250 ms -> a few ms).
    A polygon mask keeps it to motion *within* the ROI. YOLO runs on the full-res frame
    (`--work-max` 0; ~15 ms on the GPU — it resizes to --imgsz internally so 4K vs 1080 is
    detection-identical). Saved images are always full-res. Boxes are stored NORMALIZED (0..1).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("Cannot open %s", video_path)
        return None, None, None, None
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps != fps:  # 0 or NaN
        fps = 20.0

    # ROI crop box (full-res) + small padding, like detect_motion's crop_x1..crop_y2.
    if roi_pts is not None:
        pts_full = fit_polygon_to_frame(roi_pts.copy(), W, H).astype(np.int32)
        pad = max(8, int(round(0.01 * max(W, H))))
        cx1 = max(0, int(pts_full[:, 0].min()) - pad)
        cy1 = max(0, int(pts_full[:, 1].min()) - pad)
        cx2 = min(W, int(pts_full[:, 0].max()) + pad)
        cy2 = min(H, int(pts_full[:, 1].max()) + pad)
    else:
        pts_full = None
        cx1, cy1, cx2, cy2 = 0, 0, W, H
    crop_w, crop_h = max(1, cx2 - cx1), max(1, cy2 - cy1)

    # MOG2 resolution: downscale the CROP so its longest side <= --motion-max.
    mw, mh = _scaled_dims(crop_w, crop_h, args.motion_max)
    motion_resize = (mw, mh) != (crop_w, crop_h)
    # ROI polygon mask at motion resolution (inside the crop) -> motion *within* the ROI.
    if pts_full is not None:
        lp = pts_full.astype(np.float32)
        lp[:, 0] = (lp[:, 0] - cx1) * (mw / float(crop_w))
        lp[:, 1] = (lp[:, 1] - cy1) * (mh / float(crop_h))
        roi_mask = np.zeros((mh, mw), np.uint8)
        cv2.fillPoly(roi_mask, [lp.astype(np.int32)], 255)
        roi_area = max(1, int(cv2.countNonZero(roi_mask)))
    else:
        roi_mask = None
        roi_area = mw * mh

    yw, yh = _scaled_dims(W, H, args.work_max)  # YOLO frame — full by default
    yolo_resize = (yw, yh) != (W, H)

    mog = cv2.createBackgroundSubtractorMOG2(history=args.mog_history, varThreshold=args.mog_var, detectShadows=False)
    records = []
    idx = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % args.scan_stride != 0:
            continue
        crop = frame[cy1:cy2, cx1:cx2]
        mframe = cv2.resize(crop, (mw, mh), interpolation=cv2.INTER_AREA) if motion_resize else crop
        fg = mog.apply(mframe)
        if roi_mask is not None:
            fg = cv2.bitwise_and(fg, fg, mask=roi_mask)
        motion_frac = cv2.countNonZero(fg) / float(roi_area)
        if idx < args.warmup:
            continue  # MOG2 not yet stabilised
        boxes = []
        if not (args.yolo_on_motion and motion_frac < MOTION_FRAC_THRESH):
            yframe = cv2.resize(frame, (yw, yh), interpolation=cv2.INTER_AREA) if yolo_resize else frame
            res = model.predict(yframe, imgsz=args.imgsz, conf=SCAN_CONF, classes=DETECT_CLASSES, verbose=False)
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                xyxyn = res[0].boxes.xyxyn.cpu().numpy()  # normalized 0..1
                cls = res[0].boxes.cls.int().cpu().numpy()
                conf = res[0].boxes.conf.float().cpu().numpy()
                for b, c, cf in zip(xyxyn, cls, conf):
                    boxes.append((int(c), float(cf), [float(x) for x in b]))
        records.append({"idx": idx, "motion_frac": float(motion_frac), "boxes": boxes, "dhash": dhash(mframe)})
    cap.release()
    return (W, H), records, fps, ((mw, mh), (yw, yh), (crop_w, crop_h))


def mark_events(records, fps, args):
    """Flag records inside a motion-in-ROI event: motion frames, gap-merged, padded ±pad_seconds.

    Sets r['in_event'] on every record and returns the list of (start_idx, end_idx) raw events.
    This is what restricts frame selection to "motion in the ROI ± a few seconds".
    """
    pad = int(round(args.pad_seconds * fps))
    max_gap = int(round(args.max_gap_seconds * fps))
    motion_idxs = [r["idx"] for r in records if r["motion_frac"] >= MOTION_FRAC_THRESH]
    events = []
    if motion_idxs:
        s = e = motion_idxs[0]
        for x in motion_idxs[1:]:
            if x - e <= max_gap:
                e = x
            else:
                events.append((s, e)); s = e = x
        events.append((s, e))
    padded = [(a - pad, b + pad) for (a, b) in events]
    for r in records:
        r["in_event"] = any(a <= r["idx"] <= b for (a, b) in padded)
    return events


def score_records(records, args):
    """Compute active-learning hardness per record + reason flags (in place)."""
    n = len(records)
    present = []
    for r in records:
        pc = sum(1 for (c, cf, _) in r["boxes"] if c == 0 and cf >= PRESENT_CONF)
        cc = sum(1 for (c, cf, _) in r["boxes"] if c == 1 and cf >= PRESENT_CONF)
        unc = sum(1 for (c, cf, _) in r["boxes"] if SCAN_CONF <= cf < PRESENT_CONF)
        r["_person"] = pc
        r["_car"] = cc
        r["_uncertain"] = unc
        present.append(pc > 0)

    # False-negative gaps: present -> short run-of-absent -> present (mirrors flicker extractor).
    fn_gap = [False] * n
    last_present = None
    run = []
    for i, p in enumerate(present):
        if p:
            if last_present is not None and 0 < len(run) <= args.max_gap:
                for j in run:
                    fn_gap[j] = True
            run = []
            last_present = i
        elif last_present is not None:
            run.append(i)
    # a trailing absent run (person left for good) is intentionally NOT marked.

    for i, r in enumerate(records):
        prev_p = records[i - 1]["_person"] if i > 0 else r["_person"]
        next_p = records[i + 1]["_person"] if i < n - 1 else r["_person"]
        flicker = abs(r["_person"] - prev_p) + abs(r["_person"] - next_p)
        active = r["motion_frac"] >= MOTION_FRAC_THRESH
        confident = (r["_person"] + r["_car"]) > 0
        fn_motion = active and not confident
        reasons = []
        score = 0.0
        if fn_gap[i]:
            score += W_FN_GAP
            reasons.append("fn_gap")
        if fn_motion:
            score += W_FN_MOTION
            reasons.append("fn_motion")
            if r["_uncertain"] == 0:
                score += W_FN_MOTION_EMPTY
                reasons.append("clean_miss")
        if r["_uncertain"] > 0:
            score += W_UNCERTAIN * r["_uncertain"]
            reasons.append(f"uncertain:{r['_uncertain']}")
        if flicker > 0:
            score += W_FLICKER * flicker
            reasons.append(f"flicker:{flicker}")
        score += W_MOTION_TIE * min(1.0, r["motion_frac"] / max(MOTION_FRAC_THRESH, 1e-6))
        r["score"] = score
        r["reasons"] = reasons


def select_frames(records, args):
    """Pick up to per_video frames.

    select=hard   -> active learning: hardest frames first (training set).
    select=even   -> evenly spaced content frames (representative validation set).
    select=random -> random content frames, seeded (representative validation set).
    All modes honour --min-spacing and the optional visual dedup.
    """
    def pick_from(pool, selected):
        for r in pool:
            idx = r["idx"]
            if any(abs(idx - s["idx"]) < args.min_spacing for s in selected):
                continue
            if args.dhash_thresh >= 0 and any(hamming(r["dhash"], s["dhash"]) <= args.dhash_thresh for s in selected):
                continue
            selected.append(r)
            if len(selected) >= args.per_video:
                break

    # Restrict to frames inside a motion-in-ROI event (± pad). Fall back to all records only if
    # no event was detected at all (e.g. ROI disabled or a truly static clip), so we never return 0.
    base = [r for r in records if r.get("in_event")] or records

    # "content" = a frame worth labelling at all (something detected, or motion in the gate area).
    content = sorted((r for r in base
                      if (r["_person"] + r["_car"]) > 0 or r["motion_frac"] >= MOTION_FRAC_THRESH),
                     key=lambda r: r["idx"])

    selected = []
    topped_up = 0
    if args.select == "hard":
        hard = sorted((r for r in base if r["score"] > 0), key=lambda r: r["score"], reverse=True)
        pick_from(hard, selected)
        pool_n = len(hard)
        if len(selected) < args.per_video and args.topup == "motion":
            before = len(selected)
            pick_from(sorted(base, key=lambda r: r["motion_frac"], reverse=True), selected)
            topped_up = len(selected) - before
    elif args.select == "even":
        pool_n = len(content)
        cand = content
        if len(cand) > args.per_video:  # thin to ~per_video evenly across the timeline first
            step = len(cand) / args.per_video
            cand = [cand[int(i * step)] for i in range(args.per_video)]
        pick_from(cand, selected)
    else:  # random
        import random
        pool_n = len(content)
        cand = list(content)
        random.Random(args.seed).shuffle(cand)
        pick_from(cand, selected)

    selected.sort(key=lambda r: r["idx"])
    return selected, pool_n, topped_up


def save_selected(read_path, orig_name, selected, out_dir, args):
    """Second pass: decode again, save native-res frames + YOLO-format labels for selected idxs."""
    stem = os.path.splitext(orig_name)[0]
    images_dir = os.path.join(out_dir, "images")
    labels_dir = os.path.join(out_dir, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    preview_dir = os.path.join(out_dir, "preview")
    if args.preview:
        os.makedirs(preview_dir, exist_ok=True)

    want = {r["idx"]: r for r in selected}
    cap = cv2.VideoCapture(read_path)
    idx = -1
    saved = []
    while want:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        rec = want.pop(idx, None)
        if rec is None:
            continue
        name = f"{stem}_frame_{idx}"
        cv2.imwrite(os.path.join(images_dir, name + ".jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
        lines = []
        for (c, cf, (nx1, ny1, nx2, ny2)) in rec["boxes"]:  # boxes already normalized 0..1
            if cf < args.export_conf or c not in DETECT_CLASSES:
                continue
            cx = (nx1 + nx2) / 2
            cy = (ny1 + ny2) / 2
            bw = nx2 - nx1
            bh = ny2 - ny1
            cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
            bw, bh = min(max(bw, 0), 1), min(max(bh, 0), 1)
            lines.append(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(labels_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if args.preview:
            _write_preview(frame, rec, os.path.join(preview_dir, name + ".jpg"))
        saved.append({"video": orig_name, "frame": idx, "image": name + ".jpg",
                      "boxes_exported": len(lines), "score": round(rec["score"], 2), "reasons": rec["reasons"]})
    cap.release()
    return saved


def _write_preview(frame, rec, path, max_w=1280):
    img = frame.copy()
    H0, W0 = img.shape[:2]
    for (c, cf, (nx1, ny1, nx2, ny2)) in rec["boxes"]:  # normalized -> pixels
        x1, y1, x2, y2 = int(nx1 * W0), int(ny1 * H0), int(nx2 * W0), int(ny2 * H0)
        color = (100, 200, 0) if c == 0 else (200, 120, 0)
        if cf < PRESENT_CONF:
            color = (60, 60, 220)  # uncertain → red-ish
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{CLASS_NAMES[c]} {cf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(img, f"score={rec['score']:.1f} {','.join(rec['reasons'])}", (10, 40),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    scale = max_w / img.shape[1]
    if scale < 1:
        img = cv2.resize(img, (max_w, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def gather_videos(inputs):
    vids = []
    for item in inputs:
        item = os.path.abspath(item)
        if os.path.isdir(item):
            vids.extend(sorted(glob.glob(os.path.join(item, "*.mp4"))))
        elif item.lower().endswith(".mp4") and os.path.exists(item):
            vids.append(item)
        elif item.lower().endswith(".txt") and os.path.exists(item):
            with open(item) as f:
                vids.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
        else:
            logger.warning("Skipping (not a dir/.mp4/.txt list): %s", item)
    return vids


def cmd_extract(args):
    videos = gather_videos(args.inputs)
    if args.limit:
        videos = videos[: args.limit]
    if not videos:
        logger.error("No input videos found.")
        return 1
    os.makedirs(args.out, exist_ok=True)
    use_roi = args.roi.lower() != "none"
    roi_pts = load_roi_polygon(resolve_path(args.roi)) if use_roi else None
    logger.info("ROI: %s", resolve_path(args.roi) if use_roi else "full frame (motion+dedup over the whole frame)")
    model = load_model(args.model)

    manifest_path = os.path.join(args.out, "manifest.jsonl")
    total_saved = 0
    summary = []
    logger.info("Processing %d video(s) -> %s", len(videos), args.out)
    for vi, video in enumerate(videos, 1):
        t0 = time.time()
        orig_name = os.path.basename(video)
        # Optionally copy the clip to local disk once (mirrors the worker pipeline) so the two
        # decode passes hit local storage instead of the NAS twice. Default: read straight from NAS.
        read_path = video
        tmp_copy = None
        if args.copy_local and not video.startswith("/tmp/"):
            try:
                import shutil, tempfile
                fd, tmp_copy = tempfile.mkstemp(suffix=os.path.splitext(orig_name)[1] or ".mp4", prefix="dsb_")
                os.close(fd)
                shutil.copy2(video, tmp_copy)
                read_path = tmp_copy
            except Exception as e:
                logger.warning("[%d/%d] copy-local failed for %s (%s); reading from source", vi, len(videos), orig_name, e)
                read_path = video
                tmp_copy = None
        try:
            (W, H), records, fps, work = scan_video(read_path, model, roi_pts, args)
            if not records:
                logger.info("[%d/%d] %s: no scannable frames", vi, len(videos), orig_name)
                continue
            if vi == 1:
                (mw, mh), (yw, yh), (cw, ch) = work
                logger.info("Source %dx%d | ROI crop %dx%d -> MOG2 %dx%d | YOLO %dx%d%s | images saved full-res",
                            W, H, cw, ch, mw, mh, yw, yh, "" if (yw, yh) == (W, H) else " (downscaled)")
            score_records(records, args)
            events = mark_events(records, fps, args)
            ev_frames = sum(1 for r in records if r.get("in_event"))
            selected, pool_n, topped = select_frames(records, args)
            if args.dry_run:
                saved = [{"video": orig_name, "frame": r["idx"], "score": round(r["score"], 2),
                          "in_event": r.get("in_event", False), "reasons": r["reasons"]} for r in selected]
            else:
                saved = save_selected(read_path, orig_name, selected, args.out, args)
            with open(manifest_path, "a") as mf:
                for s in saved:
                    mf.write(json.dumps(s) + "\n")
            total_saved += len(saved)
            summary.append((orig_name, len(records), len(events), ev_frames, len(selected)))
            logger.info("[%d/%d] %-42s scanned=%d events=%d ev_frames=%d selected=%d%s (%.1fs)",
                        vi, len(videos), orig_name[:42], len(records), len(events), ev_frames, len(selected),
                        f" (+{topped} top-up)" if topped else "", time.time() - t0)
        finally:
            if tmp_copy and os.path.exists(tmp_copy):
                try:
                    os.remove(tmp_copy)
                except OSError:
                    pass

    print("\n=== extract summary ===")
    print(f"{'video':<46} {'scanned':>8} {'events':>7} {'ev_frames':>10} {'picked':>7}")
    for name, sc, ev, evf, pk in summary:
        print(f"{name[:46]:<46} {sc:>8} {ev:>7} {evf:>10} {pk:>7}")
    print(f"\nTotal frames {'(dry-run, not written)' if args.dry_run else 'saved'}: {total_saved}")
    print(f"Manifest: {manifest_path}")
    if not args.dry_run:
        print(f"Images:   {os.path.join(args.out, 'images')}")
        print(f"Labels:   {os.path.join(args.out, 'labels')}")
        print(f"\nNext:  python {os.path.relpath(__file__, REPO_ROOT)} pack {args.out} --out {args.out.rstrip('/')}_cvat.zip")
    return 0


# --------------------------------------------------------------------------- #
# pack
# --------------------------------------------------------------------------- #
def cmd_pack(args):
    staging = os.path.abspath(args.staging)
    images_dir = os.path.join(staging, "images")
    labels_dir = os.path.join(staging, "labels")
    if not os.path.isdir(images_dir):
        logger.error("No images/ dir in %s — run `extract` first.", staging)
        return 1
    images = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not images:
        logger.error("No images found in %s", images_dir)
        return 1
    out_zip = args.out or (staging.rstrip("/") + "_cvat.zip")

    # CVAT "YOLO 1.1" dataset layout: obj.names, obj.data, train.txt, obj_train_data/<img>.jpg + <img>.txt
    train_lines = []
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("obj.names", "\n".join(CLASS_NAMES) + "\n")
        z.writestr("obj.data", f"classes = {len(CLASS_NAMES)}\ntrain = train.txt\nnames = obj.names\nbackup = backup/\n")
        for img in images:
            base = os.path.basename(img)
            stem = os.path.splitext(base)[0]
            arc_img = f"obj_train_data/{base}"
            z.write(img, arc_img)
            train_lines.append(arc_img)
            label = os.path.join(labels_dir, stem + ".txt")
            # An empty/missing label is valid YOLO (a frame with no objects) — write empty so CVAT sees the frame.
            z.writestr(f"obj_train_data/{stem}.txt", open(label).read() if os.path.exists(label) else "")
        z.writestr("train.txt", "\n".join(train_lines) + "\n")

    n_labeled = sum(1 for img in images
                    if os.path.exists(os.path.join(labels_dir, os.path.splitext(os.path.basename(img))[0] + ".txt"))
                    and os.path.getsize(os.path.join(labels_dir, os.path.splitext(os.path.basename(img))[0] + ".txt")) > 0)
    print(f"Wrote {out_zip}")
    print(f"  images: {len(images)}  (with pre-annotations: {n_labeled})")
    print(f"  classes: {CLASS_NAMES}")
    print("\nImport in cvat.ai:")
    print("  A) Create task by uploading the images/ folder, then Actions -> Upload annotations -> 'YOLO 1.1' -> this zip")
    print("  B) or Projects/Tasks -> Create from dataset -> format 'YOLO 1.1' -> this zip (images + labels in one shot)")
    print("  Then validate/fix boxes and export.")
    return 0


# --------------------------------------------------------------------------- #
# list-crossings  (validation-set source)
# --------------------------------------------------------------------------- #
def _api_get(base, path):
    url = base.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def _daterange(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def cmd_list_crossings(args):
    if args.to:
        start = datetime.strptime(args.from_, "%Y-%m-%d").date()
        end = datetime.strptime(args.to, "%Y-%m-%d").date()
    elif args.from_:
        start = end = datetime.strptime(args.from_, "%Y-%m-%d").date()
    else:
        start = end = date.today()
    statuses = set(s.strip() for s in args.status.split(","))
    cam_root = os.path.abspath(args.camera_root)
    import random
    rng = random.Random(args.seed)

    # Videos to skip (by basename) — pass the train list here when sampling val, to keep them disjoint.
    exclude = set()
    for ex in (args.exclude or []):
        if os.path.exists(ex):
            with open(ex) as f:
                exclude.update(os.path.basename(line.strip()) for line in f if line.strip() and not line.startswith("#"))
        else:
            logger.warning("--exclude file not found: %s", ex)
    if exclude:
        logger.info("Excluding %d already-claimed video(s)", len(exclude))

    found = []
    for day in _daterange(start, end):
        day_str = day.strftime("%Y-%m-%d")
        try:
            data = _api_get(args.api, f"/api/today/videos?day={day_str}")
        except Exception as e:
            logger.warning("API fetch failed for %s: %s", day_str, e)
            continue
        vids = data.get("videos", [])
        day_dir = os.path.join(cam_root, day.strftime("%Y"), day.strftime("%m"), day.strftime("%d"))
        # Resolve every matching (non-excluded) basename to a real file path, then sample per day.
        day_paths = []
        for v in vids:
            if v.get("status") not in statuses:
                continue
            base = v.get("basename")
            if not base or base in exclude:
                continue
            cand = os.path.join(day_dir, base)
            if os.path.exists(cand):
                day_paths.append(cand)
            else:
                matches = glob.glob(os.path.join(cam_root, "**", base), recursive=True)
                if matches:
                    day_paths.append(matches[0])
                else:
                    logger.warning("%s: %s listed but file not found under %s", day_str, base, cam_root)
        total = len(day_paths)
        if args.per_day and total > args.per_day:
            day_paths = rng.sample(sorted(day_paths), args.per_day)  # sorted -> deterministic with seed
        found.extend(day_paths)
        logger.info("%s: %d matching %s (after exclude) -> kept %d", day_str, total, statuses, len(day_paths))

    found = sorted(set(found))
    if args.out:
        with open(args.out, "w") as f:
            f.write("\n".join(found) + ("\n" if found else ""))
        print(f"Wrote {len(found)} path(s) to {args.out}")
        print(f"Next:  python {os.path.relpath(__file__, REPO_ROOT)} extract {args.out} --out <val_staging> ...")
    else:
        for p in found:
            print(p)
        print(f"\n# {len(found)} video(s)", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="Extract + pre-annotate the hardest frames from videos.")
    e.add_argument("inputs", nargs="+", help="Video file(s), folder(s) of .mp4, or a .txt list of paths.")
    e.add_argument("--out", required=True, help="Staging output dir (use /mnt/nas/training/staging/<batch>, NOT temp/).")
    e.add_argument("--model", default=DEFAULT_MODEL, help=f"YOLO model (default {DEFAULT_MODEL}; pass models/yolo12s.engine on the worker).")
    e.add_argument("--roi", default=DEFAULT_ROI, help="ROI json — motion is detected WITHIN this polygon (cropped + "
                   "downscaled for speed, like detect_motion). Default config/roi-4k.json; pass 'none' for full-frame motion.")
    e.add_argument("--pad-seconds", type=float, default=3.0,
                   help="Keep frames within this many seconds before/after ROI motion (the event window). Default 3.0.")
    e.add_argument("--max-gap-seconds", type=float, default=3.0,
                   help="Merge ROI-motion bursts separated by less than this into one event. Default 3.0.")
    e.add_argument("--per-video", type=int, default=25, help="Target frames per video (default 25; use a few, e.g. 4, for validation).")
    e.add_argument("--select", choices=["hard", "even", "random"], default="hard",
                   help="Frame picker: 'hard' = active-learning hardest frames (TRAINING); 'even'/'random' = representative "
                        "content frames (VALIDATION — don't bias val toward the model's failures).")
    e.add_argument("--seed", type=int, default=0, help="RNG seed for --select random.")
    e.add_argument("--copy-local", action="store_true",
                   help="Copy each clip to local /tmp once before the two decode passes (mirrors the worker; "
                        "halves NAS reads for big batches). Default: read straight from NAS.")
    e.add_argument("--scan-stride", type=int, default=2, help="Scan every Nth frame (1=every frame; raise to go faster on CPU).")
    e.add_argument("--min-spacing", type=int, default=8, help="Min frame gap between selected frames.")
    e.add_argument("--dhash-thresh", type=int, default=-1, help="Visual near-duplicate dedup: drop a frame within this "
                   "Hamming dist of a picked one. Default -1 = OFF (diversity comes from --min-spacing; the scoring already "
                   "excludes static frames). A whole-frame hash can't see small/distant movers, so only enable (e.g. 8) for "
                   "scenes with long identical stretches.")
    e.add_argument("--max-gap", type=int, default=30, help="Max records in a present->absent->present gap to treat as false-negatives.")
    e.add_argument("--topup", choices=["none", "motion"], default="none", help="If fewer hard frames than target, top up with high-motion frames.")
    e.add_argument("--export-conf", type=float, default=PRESENT_CONF, help="Min confidence for a box to be written as a pre-annotation.")
    e.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                   help="YOLO inference size (default 640 = engine size). This — not the input frame — sets detection "
                        "detail. Raise for small/distant objects (needs an engine exported at that size).")
    e.add_argument("--motion-max", type=int, default=960,
                   help="Longest side (px) of the throwaway frame MOG2 runs on (default 960). MOG2 on full 4K is ~250ms "
                        "CPU/frame and dominates runtime; motion only needs coarse blobs, so this is the main speed knob. "
                        "Does NOT affect YOLO input or saved images. 0 = run MOG2 on the full frame (slow).")
    e.add_argument("--work-max", type=int, default=0,
                   help="Longest side (px) for the YOLO inference frame; 0 = full resolution (default). YOLO resizes to "
                        "--imgsz internally, so this is detection-neutral; saved images are ALWAYS full-res regardless.")
    e.add_argument("--yolo-on-motion", action="store_true",
                   help="Only run YOLO on frames with motion in the gate/ROI (skips dead frames). Fewer GPU calls, same "
                        "quality on the frames that matter.")
    e.add_argument("--jpeg-quality", type=int, default=92)
    e.add_argument("--mog-history", type=int, default=200)
    e.add_argument("--mog-var", type=float, default=25.0)
    e.add_argument("--warmup", type=int, default=15, help="Skip scoring the first N frames (MOG2 warmup).")
    e.add_argument("--limit", type=int, default=0, help="Process only the first N videos (smoke test).")
    e.add_argument("--preview", action="store_true", help="Also write downscaled annotated previews to <out>/preview/.")
    e.add_argument("--dry-run", action="store_true", help="Score + select only; write manifest, no images/labels.")
    e.set_defaults(func=cmd_extract)

    k = sub.add_parser("pack", help="Package a staging dir into a CVAT YOLO 1.1 zip.")
    k.add_argument("staging", help="Staging dir produced by `extract`.")
    k.add_argument("--out", help="Output zip path (default <staging>_cvat.zip).")
    k.set_defaults(func=cmd_pack)

    c = sub.add_parser("list-crossings", help="List gate_crossing video paths for a date range (val source).")
    c.add_argument("--api", default="http://localhost:8192", help="Dashboard base URL (default http://localhost:8192).")
    c.add_argument("--from", dest="from_", help="Start date YYYY-MM-DD (default today).")
    c.add_argument("--to", help="End date YYYY-MM-DD (inclusive).")
    c.add_argument("--status", default="gate_crossing", help="Comma-separated statuses to keep (default gate_crossing).")
    c.add_argument("--per-day", type=int, default=0, help="Randomly sample this many videos PER DAY (0 = all).")
    c.add_argument("--seed", type=int, default=0, help="RNG seed for --per-day sampling (reproducible).")
    c.add_argument("--exclude", action="append", metavar="LIST.txt",
                   help="Skip videos whose basename appears in this list file (repeatable). Pass the TRAIN list when "
                        "sampling VAL so the two never share a video (no duplicate frames, no leakage).")
    c.add_argument("--camera-root", default="/mnt/nas/xiaomi_camera_videos/04cf8c6b201d/", help="Root of YYYY/MM/DD camera videos.")
    c.add_argument("--out", help="Write the path list to this file (else stdout).")
    c.set_defaults(func=cmd_list_crossings)
    return p


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)], force=True)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
