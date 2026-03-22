#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
from collections import defaultdict

import cv2
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from person_id import PersonReID


def _default_model_path(repo_root: str) -> str:
    return os.path.join(
        repo_root,
        "models",
        "reid",
        "intel",
        "person-reidentification-retail-0288",
        "FP16",
        "person-reidentification-retail-0288.xml",
    )


def _laplacian_sharpness(path: str) -> float:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def _safe_move(src: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    dst = os.path.join(dst_dir, base)
    if not os.path.exists(dst):
        shutil.move(src, dst)
        return dst

    stem, ext = os.path.splitext(base)
    k = 1
    while True:
        cand = os.path.join(dst_dir, f"{stem}_{k}{ext}")
        if not os.path.exists(cand):
            shutil.move(src, cand)
            return cand
        k += 1


def _draw_multiline_text(canvas: np.ndarray, lines: list[str], x: int, y: int, color=(255, 255, 255)) -> None:
    line_h = 16
    for i, t in enumerate(lines):
        cv2.putText(
            canvas,
            t,
            (x, y + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )


def _fit_image_to_tile(img: np.ndarray, tile_w: int, tile_h: int, label_h: int) -> np.ndarray:
    inner_w = max(20, tile_w - 8)
    inner_h = max(20, tile_h - label_h - 10)
    src_h, src_w = img.shape[:2]
    if src_h <= 0 or src_w <= 0:
        return np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    scale = min(float(inner_w) / float(src_w), float(inner_h) / float(src_h))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

    tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
    x0 = (tile_w - new_w) // 2
    y0 = max(4, (inner_h - new_h) // 2 + 4)
    tile[y0:y0 + new_h, x0:x0 + new_w] = resized
    return tile


def _build_group_sheet(
    group_num: int,
    idxs: list[int],
    keep_idx: int,
    paths: list[str],
    sim: np.ndarray,
    sharpness: dict[int, float],
    out_path: str,
    tile_w: int,
    tile_h: int,
    cols: int,
) -> bool:
    if len(idxs) == 0:
        return False

    label_h = 58
    cols = max(1, int(cols))
    rows = (len(idxs) + cols - 1) // cols

    header_h = 36
    pad = 8
    canvas_w = cols * tile_w + (cols + 1) * pad
    canvas_h = header_h + rows * tile_h + (rows + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 20, dtype=np.uint8)

    cv2.putText(
        canvas,
        f"Group {group_num} (n={len(idxs)}) KEEP=green",
        (pad, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )

    for pos, idx in enumerate(idxs):
        row = pos // cols
        col = pos % cols
        x1 = pad + col * (tile_w + pad)
        y1 = header_h + pad + row * (tile_h + pad)
        x2 = x1 + tile_w
        y2 = y1 + tile_h

        img = cv2.imread(paths[idx])
        if img is None:
            tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            _draw_multiline_text(tile, ["LOAD FAILED", os.path.basename(paths[idx])[:26]], 8, 24, color=(40, 60, 255))
        else:
            tile = _fit_image_to_tile(img, tile_w, tile_h, label_h)

        mark = "KEEP" if idx == keep_idx else "DROP"
        sim_keep = 1.0 if idx == keep_idx else float(sim[idx, keep_idx])
        sh = float(sharpness.get(idx, 0.0))
        name = os.path.basename(paths[idx])
        if len(name) > 30:
            name = name[:27] + "..."
        lines = [
            f"{mark} sim={sim_keep:.3f}",
            f"sharp={sh:.1f}",
            name,
        ]
        _draw_multiline_text(tile, lines, 8, tile_h - label_h + 18)

        canvas[y1:y2, x1:x2] = tile
        border_color = (60, 200, 60) if idx == keep_idx else (120, 120, 120)
        border_thick = 4 if idx == keep_idx else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, border_thick)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    return bool(cv2.imwrite(out_path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 92]))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Find near-duplicate ReID gallery crops by cosine similarity and optionally clean them up."
    )
    parser.add_argument("--gallery", required=True, help="Path to ReID gallery directory.")
    parser.add_argument(
        "--model",
        default=None,
        help="Path to OpenVINO ReID model XML. Defaults to models/reid/.../person-reidentification-retail-0288.xml",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.95,
        help="Cosine similarity threshold for marking images as near-duplicates (default: 0.95).",
    )
    parser.add_argument(
        "--min-group-size",
        type=int,
        default=2,
        help="Minimum similar-group size to report (default: 2).",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=0,
        help="Max groups to print (0 = all).",
    )
    parser.add_argument(
        "--move-duplicates-to",
        default=None,
        help="If set, move duplicate files (except one kept per group) to this directory.",
    )
    parser.add_argument(
        "--delete-duplicates",
        action="store_true",
        help="Delete duplicate files (except one kept per group). Use with care.",
    )
    parser.add_argument(
        "--viz-dir",
        default=None,
        help="If set, save group visualization contact sheets into this directory.",
    )
    parser.add_argument(
        "--viz-cols",
        type=int,
        default=6,
        help="Columns in visualization grid (default: 6). Use 0 for single-row layout.",
    )
    parser.add_argument(
        "--viz-tile-width",
        type=int,
        default=240,
        help="Visualization tile width in pixels (default: 240).",
    )
    parser.add_argument(
        "--viz-tile-height",
        type=int,
        default=280,
        help="Visualization tile height in pixels (default: 280).",
    )
    parser.add_argument(
        "--viz-max-items",
        type=int,
        default=0,
        help="Cap images rendered per group (0 = all). Useful for very large groups.",
    )
    args = parser.parse_args()

    if args.move_duplicates_to and args.delete_duplicates:
        print("ERROR: Use either --move-duplicates-to or --delete-duplicates, not both.")
        return 2

    gallery = os.path.abspath(args.gallery)
    if not os.path.isdir(gallery):
        print(f"ERROR: Gallery does not exist: {gallery}")
        return 2

    model_path = os.path.abspath(args.model) if args.model else _default_model_path(REPO_ROOT)
    if not os.path.isfile(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return 2

    reid = PersonReID(model_path=model_path, gallery_path=gallery, threshold=0.0)
    vectors = reid.gallery_vectors
    paths = reid.gallery_paths

    if len(vectors) == 0 or len(paths) == 0:
        print("No gallery images with valid embeddings found.")
        return 0

    if len(vectors) != len(paths):
        print("ERROR: Embedding/path count mismatch. Please rerun after clearing temp ReID cache.")
        return 2

    mat = np.stack(vectors).astype(np.float32)
    # Safety normalization in case vectors come from old cache.
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    mat = mat / norms

    sim = mat @ mat.T
    n = sim.shape[0]

    dsu = _DSU(n)
    pair_count = 0
    thr = float(args.similarity_threshold)
    for i in range(n):
        for j in range(i + 1, n):
            if float(sim[i, j]) >= thr:
                dsu.union(i, j)
                pair_count += 1

    groups = defaultdict(list)
    for i in range(n):
        groups[dsu.find(i)].append(i)

    similar_groups = [idxs for idxs in groups.values() if len(idxs) >= max(2, int(args.min_group_size))]
    similar_groups.sort(key=len, reverse=True)

    print(f"Gallery images: {n}")
    print(f"Similarity threshold: {thr:.3f}")
    print(f"Similar pairs: {pair_count}")
    print(f"Groups >= {max(2, int(args.min_group_size))}: {len(similar_groups)}")

    if not similar_groups:
        return 0

    viz_saved = 0
    viz_failed = 0
    viz_dir = os.path.abspath(args.viz_dir) if args.viz_dir else None
    if viz_dir:
        os.makedirs(viz_dir, exist_ok=True)

    moved = 0
    deleted = 0

    max_groups = int(args.max_groups)
    to_show = similar_groups if max_groups <= 0 else similar_groups[:max_groups]

    for g_idx, idxs in enumerate(to_show, start=1):
        sharpness = {i: _laplacian_sharpness(paths[i]) for i in idxs}
        sharp = list(sharpness.items())
        sharp.sort(key=lambda x: x[1], reverse=True)
        keep_idx = sharp[0][0]

        idxs_for_output = idxs
        if int(args.viz_max_items) > 0 and len(idxs_for_output) > int(args.viz_max_items):
            idxs_for_output = list(idxs_for_output)
            if keep_idx not in idxs_for_output[: int(args.viz_max_items)]:
                idxs_for_output = [keep_idx] + [i for i in idxs_for_output if i != keep_idx]
            idxs_for_output = idxs_for_output[: int(args.viz_max_items)]

        if viz_dir:
            cols = int(args.viz_cols)
            if cols <= 0:
                cols = max(1, len(idxs_for_output))
            out_path = os.path.join(viz_dir, f"group_{g_idx:03d}.jpg")
            ok = _build_group_sheet(
                group_num=g_idx,
                idxs=idxs_for_output,
                keep_idx=keep_idx,
                paths=paths,
                sim=sim,
                sharpness=sharpness,
                out_path=out_path,
                tile_w=max(120, int(args.viz_tile_width)),
                tile_h=max(160, int(args.viz_tile_height)),
                cols=cols,
            )
            if ok:
                viz_saved += 1
                print(f"  viz -> {out_path}")
            else:
                viz_failed += 1
                print(f"  viz failed -> {out_path}")

        print("-" * 80)
        print(f"Group #{g_idx}: size={len(idxs)}; keep={os.path.basename(paths[keep_idx])}")

        for i in idxs:
            mark = "KEEP" if i == keep_idx else "DROP"
            s_keep = float(sim[i, keep_idx]) if i != keep_idx else 1.0
            print(f"  [{mark}] sim_to_keep={s_keep:.4f} sharp={sharpness[i]:.2f} path={paths[i]}")

        if args.move_duplicates_to or args.delete_duplicates:
            for i in idxs:
                if i == keep_idx:
                    continue
                p = paths[i]
                if args.move_duplicates_to:
                    dst = _safe_move(p, os.path.abspath(args.move_duplicates_to))
                    moved += 1
                    print(f"    moved -> {dst}")
                elif args.delete_duplicates:
                    try:
                        os.remove(p)
                        deleted += 1
                        print(f"    deleted -> {p}")
                    except Exception as e:
                        print(f"    failed delete -> {p}: {e}")

    if args.move_duplicates_to:
        print(f"Moved duplicates: {moved}")
    if args.delete_duplicates:
        print(f"Deleted duplicates: {deleted}")
    if viz_dir:
        print(f"Visualizations saved: {viz_saved}, failed: {viz_failed}, dir={viz_dir}")

    if max_groups > 0 and len(similar_groups) > max_groups:
        print(f"Note: displayed {max_groups} groups out of {len(similar_groups)}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
