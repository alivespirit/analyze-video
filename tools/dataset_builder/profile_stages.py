#!/usr/bin/env python3
"""Profile per-frame stage costs ON THE WORKER (real GPU + TensorRT engine).

Shows exactly where build_dataset.py's time goes — decode and MOG2 (CPU, the GPU can't
help) vs YOLO predict (GPU) — at full 4K and at a downscaled working size, so we fix the
real bottleneck instead of guessing across different hardware.

Run on the worker, from the repo root:

  python tools/dataset_builder/profile_stages.py \
      /mnt/nas/analyze-video/temp/training/Balcony-00-124041-124057.mp4 \
      --model models/yolo12s.engine --frames 120 --work-max 1920

Read the output like this:
  * YOLO_predict_*(GPU) small  -> the GPU is fine; wall-time is CPU-bound (decode/MOG2)
  * MOG2_full large            -> MOG2 on 4K is the hog (it's CPU; shrink its input or gate it)
  * predict_FULL ~= predict_WORK -> feeding YOLO 4K vs 1080 is the same (it resizes to imgsz anyway)
"""
import argparse
import collections
import os
import sys
import time

import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--model", default=os.path.join("models", "yolo12s.engine"))
    ap.add_argument("--frames", type=int, default=120, help="Frames to time (after warmup).")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--work-max", type=int, default=1920, help="Longest side of the downscaled comparison frame; 0 = skip it.")
    ap.add_argument("--conf", type=float, default=0.10)
    args = ap.parse_args()

    model_path = args.model if os.path.exists(args.model) else os.path.join(REPO_ROOT, args.model)
    from ultralytics import YOLO
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "CPU"
    except Exception:
        torch, cuda, dev = None, False, "unknown"
    print(f"Device: {dev}  (torch.cuda={cuda})")
    print(f"Model:  {model_path}")
    model = YOLO(model_path, task="detect")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("cannot open", args.video)
        return 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    longest = max(W, H)
    sw, sh = W, H
    if args.work_max and longest > args.work_max:
        sc = args.work_max / float(longest)
        sw, sh = int(round(W * sc)), int(round(H * sc))
    downscale = (sw, sh) != (W, H)
    print(f"Source: {W}x{H}   work: {sw}x{sh}   imgsz: {args.imgsz}\n")

    def sync():
        if cuda:
            torch.cuda.synchronize()

    def predict(img):
        sync()
        t = time.perf_counter()
        model.predict(img, imgsz=args.imgsz, conf=args.conf, classes=[0, 1], verbose=False)
        sync()
        return time.perf_counter() - t

    ok, f0 = cap.read()
    if not ok:
        print("no frames")
        return 1
    s0 = cv2.resize(f0, (sw, sh), interpolation=cv2.INTER_AREA) if downscale else f0
    print("warming up TensorRT (first calls build the context)...")
    for _ in range(8):
        predict(f0)
        if downscale:
            predict(s0)

    mog_full = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    mog_work = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    T = collections.defaultdict(float)
    N = 0
    cap.release()
    cap = cv2.VideoCapture(args.video)  # restart at frame 0
    while N < args.frames:
        t = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        T["decode"] += time.perf_counter() - t

        t = time.perf_counter(); mog_full.apply(frame); T[f"MOG2_full_{W}x{H}"] += time.perf_counter() - t

        small = frame
        if downscale:
            t = time.perf_counter(); small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA); T["resize_to_work"] += time.perf_counter() - t
            t = time.perf_counter(); mog_work.apply(small); T[f"MOG2_work_{sw}x{sh}"] += time.perf_counter() - t

        T["YOLO_predict_FULL(GPU)"] += predict(frame)
        if downscale:
            T["YOLO_predict_WORK(GPU)"] += predict(small)
        N += 1
    cap.release()

    print(f"\nsampled {N} frames\n")
    for k in sorted(T, key=lambda x: -T[x]):
        print(f"  {k:28s} {1000 * T[k] / N:8.1f} ms/frame")

    full = (T["decode"] + T[f"MOG2_full_{W}x{H}"] + T["YOLO_predict_FULL(GPU)"]) / N
    print(f"\n  current path (full-res MOG2 + predict): {1000 * full:6.0f} ms/frame  -> {full:.2f} s x N frames/video")
    if downscale:
        work = (T["decode"] + T["resize_to_work"] + T[f"MOG2_work_{sw}x{sh}"] + T["YOLO_predict_WORK(GPU)"]) / N
        print(f"  work-frame path (MOG2 on {sw}x{sh}):       {1000 * work:6.0f} ms/frame  ({full / max(work, 1e-9):.1f}x faster, same detection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
