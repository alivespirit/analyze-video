#!/usr/bin/env python3
"""Draw YOLO label boxes onto dataset images for visual QA (find mislabeled frames).

Reads <dataset>/{images,labels}/<split>/, draws each label's bbox on its image, and writes
previews to an output dir (mirroring the split). Filter by file mtime to review only older
annotations — e.g. everything before a known-good change.

  python tools/dataset_builder/preview_labels.py \
      --dataset /mnt/nas/training/datasets --out /mnt/nas/training/label_previews \
      --before 2026-06-14

Boxes only by default (person=green, car=blue); pass --label-text to also draw class names.
"""
import argparse
import glob
import os
from datetime import datetime

import cv2

CLASS_COLORS = {0: (80, 200, 0), 1: (220, 120, 0)}  # BGR: person=green, car=blue
CLASS_NAMES = {0: "person", 1: "car"}
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="/mnt/nas/training/datasets")
    ap.add_argument("--out", default="/mnt/nas/training/label_previews")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--before", help="Only images with mtime BEFORE this date (YYYY-MM-DD).")
    ap.add_argument("--after", help="Only images with mtime on/after this date (YYYY-MM-DD).")
    ap.add_argument("--by", choices=["image", "label"], default="image", help="Which file's mtime to filter on (default image).")
    ap.add_argument("--max-dim", type=int, default=1920, help="Downscale preview longest side for easy browsing (0 = full res).")
    ap.add_argument("--label-text", action="store_true", help="Also draw the class name above each box.")
    ap.add_argument("--only-labeled", action="store_true", help="Skip images with no/empty label file.")
    ap.add_argument("--limit", type=int, default=0, help="Process at most N images (smoke test).")
    args = ap.parse_args()

    before = datetime.strptime(args.before, "%Y-%m-%d").timestamp() if args.before else None
    after = datetime.strptime(args.after, "%Y-%m-%d").timestamp() if args.after else None

    total = drawn = nolabel = boxes = 0
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        img_dir = os.path.join(args.dataset, "images", split)
        lbl_dir = os.path.join(args.dataset, "labels", split)
        out_dir = os.path.join(args.out, split)
        os.makedirs(out_dir, exist_ok=True)
        for img_path in sorted(p for p in glob.glob(os.path.join(img_dir, "*")) if p.lower().endswith(IMG_EXTS)):
            stem = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            ref = lbl_path if (args.by == "label" and os.path.exists(lbl_path)) else img_path
            mt = os.path.getmtime(ref)
            if before is not None and mt >= before:
                continue
            if after is not None and mt < after:
                continue
            has_label = os.path.exists(lbl_path) and os.path.getsize(lbl_path) > 0
            if args.only_labeled and not has_label:
                continue
            total += 1
            if args.limit and total > args.limit:
                total -= 1
                break
            img = cv2.imread(img_path)
            if img is None:
                continue
            H, W = img.shape[:2]
            th = max(2, round(max(W, H) / 600))
            if has_label:
                with open(lbl_path) as f:
                    for line in f:
                        p = line.split()
                        if len(p) < 5:
                            continue
                        c = int(float(p[0]))
                        cx, cy, bw, bh = map(float, p[1:5])
                        x1, y1 = int((cx - bw / 2) * W), int((cy - bh / 2) * H)
                        x2, y2 = int((cx + bw / 2) * W), int((cy + bh / 2) * H)
                        color = CLASS_COLORS.get(c, (255, 255, 255))
                        cv2.rectangle(img, (x1, y1), (x2, y2), color, th)
                        if args.label_text:
                            cv2.putText(img, CLASS_NAMES.get(c, str(c)), (x1, max(12, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, max(W, H) / 2200), color, th, cv2.LINE_AA)
                        boxes += 1
            else:
                nolabel += 1
            if args.max_dim and max(W, H) > args.max_dim:
                sc = args.max_dim / max(W, H)
                img = cv2.resize(img, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(out_dir, stem + ".jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            drawn += 1

    print(f"Wrote {drawn} preview(s) to {args.out}")
    print(f"  matched {total} images | {boxes} boxes drawn | {nolabel} image(s) had no label")


if __name__ == "__main__":
    raise SystemExit(main())
