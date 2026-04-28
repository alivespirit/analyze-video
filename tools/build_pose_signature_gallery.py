#!/usr/bin/env python3
import argparse
import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Ensure we can import from repo root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

from pose_signature import PoseSignatureCollector, load_pose_signature_zones  # noqa: E402

try:
    from ultralytics import YOLO  # noqa: E402
except Exception as e:
    raise RuntimeError(f"Failed to import YOLO: {e}")


def expand_video_inputs(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for raw in items:
        path = os.path.abspath(raw)
        candidates: List[str] = []

        if any(ch in raw for ch in ("*", "?", "[")):
            candidates = [os.path.abspath(p) for p in glob.glob(raw)]
        elif os.path.isdir(path):
            candidates = sorted(glob.glob(os.path.join(path, "*.mp4")))
        elif os.path.isfile(path):
            candidates = [path]

        for c in candidates:
            if c.lower().endswith(".mp4") and c not in seen:
                seen.add(c)
                out.append(c)

    out.sort()
    return out


def pick_largest_bbox(pose_result):
    try:
        boxes = pose_result.boxes.xyxy
        if boxes is None:
            return None
        if hasattr(boxes, "cpu"):
            boxes = boxes.cpu().numpy()
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            return None
        areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
        idx = int(np.argmax(areas))
        if idx < 0 or idx >= arr.shape[0]:
            return None
        b = arr[idx]
        return (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    except Exception:
        return None


def pick_best_person_bbox(det_result) -> Optional[Tuple[float, float, float, float]]:
    """Pick the most suitable person bbox from detector output.

    Uses largest-area person as a simple robust heuristic for curated
    enrollment clips where the target is typically prominent.
    """
    try:
        boxes = det_result.boxes
        if boxes is None:
            return None
        xyxy = boxes.xyxy
        cls = boxes.cls
        if xyxy is None or cls is None:
            return None
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        if hasattr(cls, "cpu"):
            cls = cls.cpu().numpy()
        arr = np.asarray(xyxy, dtype=np.float32)
        cls_arr = np.asarray(cls, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] <= 0:
            return None

        best = None
        best_area = -1.0
        for i in range(arr.shape[0]):
            # COCO class 0 = person
            if i >= cls_arr.shape[0] or int(cls_arr[i]) != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in arr[i]]
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area > best_area:
                best_area = area
                best = (x1, y1, x2, y2)
        return best
    except Exception:
        return None


def crop_with_padding(frame: np.ndarray, bbox: Tuple[float, float, float, float], padding: int):
    try:
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, int(x1) - int(padding))
        y1 = max(0, int(y1) - int(padding))
        x2 = min(int(w), int(x2) + int(padding))
        y2 = min(int(h), int(y2) + int(padding))
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return crop, (float(x1), float(y1), float(x2), float(y2))
    except Exception:
        return None


def build_embedding_for_video(
    video_path: str,
    detector_model,
    pose_model,
    *,
    imgsz: int,
    detector_imgsz: int,
    detector_conf: float,
    detector_iou: float,
    pose_conf: float,
    frame_stride: int,
    pose_on_full_frame: bool,
    crop_padding: int,
    zones,
    keypoint_conf: float,
    min_upper_kp: int,
    min_lower_kp: int,
    min_frames: int,
    smooth_alpha: float,
    max_interp_gap_frames: int,
):
    collector = PoseSignatureCollector(
        zones=zones,
        keypoint_conf_thr=keypoint_conf,
        min_upper_kp=min_upper_kp,
        min_lower_kp=min_lower_kp,
        min_frames=min_frames,
        smooth_alpha=smooth_alpha,
        max_interp_gap_frames=max_interp_gap_frames,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            if frame_stride > 1 and (idx % frame_stride) != 0:
                idx += 1
                continue

            pose_input = frame
            zone_bbox = None

            if not pose_on_full_frame:
                try:
                    det_results = detector_model.predict(
                        frame,
                        imgsz=detector_imgsz,
                        conf=detector_conf,
                        iou=detector_iou,
                        classes=[0],
                        verbose=False,
                    )
                except Exception:
                    idx += 1
                    continue

                if not det_results:
                    idx += 1
                    continue

                best_bbox = pick_best_person_bbox(det_results[0])
                if best_bbox is None:
                    idx += 1
                    continue

                cropped = crop_with_padding(frame, best_bbox, padding=crop_padding)
                if cropped is None:
                    idx += 1
                    continue

                pose_input, zone_bbox = cropped

            try:
                pose_results = pose_model.predict(
                    pose_input,
                    imgsz=imgsz,
                    conf=pose_conf,
                    verbose=False,
                )
            except Exception:
                idx += 1
                continue

            if pose_results:
                result = pose_results[0]
                if zone_bbox is None:
                    zone_bbox = pick_largest_bbox(result)
                collector.consume(result, bbox=zone_bbox)

            idx += 1
    finally:
        cap.release()

    seq = collector.finalize()
    emb = seq.get("embedding")
    return emb, seq


def load_existing_gallery(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Build or update pose-signature gallery template from curated videos."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Video files, directories, or glob patterns.",
    )
    parser.add_argument(
        "--identity",
        required=True,
        help="Identity name to create/update in the gallery.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "config", "pose_signature_gallery.json"),
        help="Output gallery JSON path.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing gallery file if it exists.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("POSE_MODEL_PATH", os.path.join(REPO_ROOT, "models", "yolo26s-pose.engine")),
        help="Pose model path.",
    )
    parser.add_argument(
        "--detector-model",
        default=os.getenv("OBJECT_DETECTION_MODEL_PATH", os.path.join(REPO_ROOT, "models", "yolo12s.engine")),
        help="Detector model path used to crop persons before pose inference.",
    )
    parser.add_argument("--imgsz", type=int, default=int(os.getenv("POSE_IMGSZ", "640")))
    parser.add_argument("--detector-imgsz", type=int, default=int(os.getenv("IMGSZ", "640")))
    parser.add_argument("--detector-conf", type=float, default=float(os.getenv("CONF_THRESHOLD", "0.35")))
    parser.add_argument("--detector-iou", type=float, default=float(os.getenv("IOU_THRESHOLD", "0.7")))
    parser.add_argument("--pose-conf", type=float, default=float(os.getenv("POSE_CONF_THRESHOLD", "0.25")))
    parser.add_argument("--crop-padding", type=int, default=int(os.getenv("POSE_CROP_PADDING", "10")))
    parser.add_argument(
        "--pose-on-full-frame",
        action="store_true",
        help="Run pose directly on full frame instead of detector-cropped person (not recommended).",
    )
    parser.add_argument("--frame-stride", type=int, default=1, help="Use every N-th frame.")
    parser.add_argument(
        "--roi-config",
        default=os.path.join(REPO_ROOT, "config", "roi.json"),
        help="ROI config path providing pose_estimation_roi (fallback: person_tracker_roi), e.g. roi.json, roi-1080p.json, roi-4k.json.",
    )
    parser.add_argument("--keypoint-conf", type=float, default=float(os.getenv("POSE_SIGNATURE_KEYPOINT_CONF", "0.25")))
    parser.add_argument("--min-upper-kp", type=int, default=int(os.getenv("POSE_SIGNATURE_MIN_UPPER_KP", "3")))
    parser.add_argument("--min-lower-kp", type=int, default=int(os.getenv("POSE_SIGNATURE_MIN_LOWER_KP", "4")))
    parser.add_argument("--min-frames", type=int, default=int(os.getenv("POSE_SIGNATURE_MIN_FRAMES", "24")))
    parser.add_argument(
        "--min-sequence-quality",
        type=float,
        default=float(os.getenv("POSE_SIGNATURE_MIN_SEQUENCE_QUALITY", "0.30")),
        help="Skip enrollment clips whose usable-frame ratio is below this threshold (0..1).",
    )
    parser.add_argument("--smooth-alpha", type=float, default=float(os.getenv("POSE_SIGNATURE_SMOOTH_ALPHA", "0.65")))
    parser.add_argument("--max-interp-gap", type=int, default=int(os.getenv("POSE_SIGNATURE_MAX_INTERP_GAP", "2")))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    args = parser.parse_args()
    args.min_sequence_quality = max(0.0, min(1.0, float(args.min_sequence_quality)))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    videos = expand_video_inputs(args.inputs)
    if not videos:
        raise SystemExit("No input videos found.")

    logging.info("Building pose-signature template for identity=%s from %d video(s).", args.identity, len(videos))

    zones = load_pose_signature_zones(
        args.roi_config,
        preferred_key="pose_estimation_roi",
        fallback_key="person_tracker_roi",
    )
    if zones.get("analyze"):
        logging.info(
            "Pose-signature zone source: %s (preferred key=pose_estimation_roi, fallback=person_tracker_roi).",
            args.roi_config,
        )
    else:
        logging.warning(
            "ROI config has no valid pose_estimation_roi or person_tracker_roi polygon (%s). Zone filtering is disabled.",
            args.roi_config,
        )

    pose_model = YOLO(args.model, task="pose")
    detector_model = None
    if not args.pose_on_full_frame:
        detector_model = YOLO(args.detector_model, task="detect")
        logging.info(
            "Enrollment mode: detector-crop (detector=%s, detector_imgsz=%d).",
            args.detector_model,
            int(args.detector_imgsz),
        )
    else:
        logging.info("Enrollment mode: full-frame pose.")

    embeddings = []
    per_video_stats = []

    for video_path in videos:
        try:
            emb, seq = build_embedding_for_video(
                video_path,
                detector_model,
                pose_model,
                imgsz=args.imgsz,
                detector_imgsz=args.detector_imgsz,
                detector_conf=args.detector_conf,
                detector_iou=args.detector_iou,
                pose_conf=args.pose_conf,
                frame_stride=max(1, int(args.frame_stride)),
                pose_on_full_frame=bool(args.pose_on_full_frame),
                crop_padding=max(0, int(args.crop_padding)),
                zones=zones,
                keypoint_conf=args.keypoint_conf,
                min_upper_kp=args.min_upper_kp,
                min_lower_kp=args.min_lower_kp,
                min_frames=args.min_frames,
                smooth_alpha=args.smooth_alpha,
                max_interp_gap_frames=max(0, int(args.max_interp_gap)),
            )
        except Exception as e:
            logging.warning("Failed processing %s: %s", video_path, e)
            continue

        frames_seen = int(seq.get("frames_seen", 0))
        frames_used = int(seq.get("frames_used", 0))
        frames_dropped_zone = int(seq.get("frames_dropped_zone", 0))
        frames_dropped_quality = int(seq.get("frames_dropped_quality", 0))
        quality = float(seq.get("sequence_quality", 0.0))

        if frames_seen > 0:
            zone_drop_pct = 100.0 * (float(frames_dropped_zone) / float(frames_seen))
            quality_drop_pct = 100.0 * (float(frames_dropped_quality) / float(frames_seen))
        else:
            zone_drop_pct = 0.0
            quality_drop_pct = 0.0

        if emb is None:
            logging.info(
                "Skip %s: insufficient usable frames (%d/%d, quality=%.2f, zone_drop=%d[%.1f%%], quality_drop=%d[%.1f%%]).",
                os.path.basename(video_path),
                frames_used,
                frames_seen,
                quality,
                frames_dropped_zone,
                zone_drop_pct,
                frames_dropped_quality,
                quality_drop_pct,
            )
            continue

        if quality < args.min_sequence_quality:
            logging.info(
                "Skip %s: low sequence quality (%.2f < %.2f, zone_drop=%d[%.1f%%], quality_drop=%d[%.1f%%]).",
                os.path.basename(video_path),
                quality,
                args.min_sequence_quality,
                frames_dropped_zone,
                zone_drop_pct,
                frames_dropped_quality,
                quality_drop_pct,
            )
            continue

        embeddings.append(np.asarray(emb, dtype=np.float32))
        per_video_stats.append(
            {
                "video": video_path,
                "frames_seen": frames_seen,
                "frames_used": frames_used,
                "sequence_quality": quality,
            }
        )
        logging.info(
            "Accepted %s: frames=%d/%d quality=%.2f (zone_drop=%d[%.1f%%], quality_drop=%d[%.1f%%])",
            os.path.basename(video_path),
            frames_used,
            frames_seen,
            quality,
            frames_dropped_zone,
            zone_drop_pct,
            frames_dropped_quality,
            quality_drop_pct,
        )

    if not embeddings:
        raise SystemExit("No usable embeddings produced. Try cleaner clips or lower min-frames/quality gates.")

    emb_stack = np.stack(embeddings, axis=0).astype(np.float32)
    mean_emb = np.mean(emb_stack, axis=0)
    norm = float(np.linalg.norm(mean_emb))
    if norm <= 1e-8:
        raise SystemExit("Failed to build template: zero-norm embedding.")
    mean_emb = mean_emb / norm

    output_path = os.path.abspath(args.output)
    payload = load_existing_gallery(output_path) if args.merge else {}
    if not isinstance(payload, dict):
        payload = {}

    identities = payload.get("identities") if isinstance(payload.get("identities"), dict) else {}
    identities[args.identity] = {
        "embedding": mean_emb.tolist(),
        "meta": {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "video_count": len(per_video_stats),
            "min_frames": int(args.min_frames),
            "min_sequence_quality": float(args.min_sequence_quality),
            "frame_stride": int(max(1, args.frame_stride)),
            "mode": "full_frame_pose" if bool(args.pose_on_full_frame) else "detector_crop_pose",
            "detector_model": args.detector_model if not bool(args.pose_on_full_frame) else None,
            "source_videos": [s["video"] for s in per_video_stats],
            "avg_frames_used": float(np.mean([s["frames_used"] for s in per_video_stats])),
            "avg_sequence_quality": float(np.mean([s["sequence_quality"] for s in per_video_stats])),
        },
    }

    payload["version"] = 1
    payload["feature_version"] = "pose_signature_v1"
    payload["identities"] = identities

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logging.info("Pose-signature gallery updated: %s", output_path)
    logging.info("Identity '%s' updated with %d sequence(s).", args.identity, len(per_video_stats))


if __name__ == "__main__":
    main()
