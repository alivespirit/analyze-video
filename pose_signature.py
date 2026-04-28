import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO keypoint indices used by YOLO pose models.
UPPER_KEYPOINTS = (5, 6, 7, 8, 9, 10)
LOWER_KEYPOINTS = (11, 12, 13, 14, 15, 16)
SIGNATURE_KEYPOINTS = UPPER_KEYPOINTS + LOWER_KEYPOINTS


def _to_polygon_list(raw: object) -> List[np.ndarray]:
    polys: List[np.ndarray] = []
    if not isinstance(raw, list):
        return polys
    for item in raw:
        try:
            arr = np.asarray(item, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2:
                continue
            polys.append(arr)
        except Exception:
            continue
    return polys


def load_pose_signature_zones(
        path: str,
        preferred_key: str = "pose_estimation_roi",
        fallback_key: str = "person_tracker_roi",
) -> Dict[str, object]:
    """Load pose-signature analyze zone from an ROI config JSON file.

        Expected ROI JSON keys:
            - pose_estimation_roi: polygon points [[x, y], ...]  # preferred
            - person_tracker_roi: polygon points [[x, y], ...]   # fallback
    """
    cfg: Dict[str, object] = {"analyze": []}

    if not path or not os.path.exists(path):
        return cfg

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return cfg

        cfg["analyze"] = _to_polygon_list([data.get(preferred_key, [])])
        if not cfg["analyze"]:
            cfg["analyze"] = _to_polygon_list([data.get(fallback_key, [])])
            if cfg["analyze"]:
                logger.info(
                    "ROI config %s: using fallback '%s' for pose signature zone (missing '%s').",
                    path,
                    fallback_key,
                    preferred_key,
                )

        if not cfg["analyze"]:
            logger.warning(
                "ROI config %s has no valid '%s' or '%s' polygon for pose signature; zone filtering is disabled.",
                path,
                preferred_key,
                fallback_key,
            )
    except Exception as e:
        logger.warning("Failed to load pose-signature zone from ROI config %s: %s", path, e)

    return cfg


def _point_in_any(point_xy: Tuple[float, float], polygons: List[np.ndarray]) -> bool:
    for poly in polygons:
        try:
            if cv2.pointPolygonTest(poly, point_xy, False) >= 0:
                return True
        except Exception:
            continue
    return False


def zone_weight_for_bbox(
    bbox: Optional[Tuple[float, float, float, float]],
    zones: Optional[Dict[str, object]],
) -> float:
    """Return per-frame recognition weight based on configured polygons."""
    if bbox is None or zones is None:
        return 1.0

    try:
        x1, y1, x2, y2 = bbox
        cx = 0.5 * (float(x1) + float(x2))
        cy = 0.5 * (float(y1) + float(y2))
    except Exception:
        return 1.0

    pt = (cx, cy)
    analyze = zones.get("analyze", [])
    if not analyze:
        return 1.0
    return 1.0 if _point_in_any(pt, analyze) else 0.0


def load_pose_signature_gallery(path: str) -> Dict[str, np.ndarray]:
    """Load identity -> normalized embedding map from gallery JSON."""
    out: Dict[str, np.ndarray] = {}
    if not path or not os.path.exists(path):
        return out

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        identities = data.get("identities", {}) if isinstance(data, dict) else {}
        if not isinstance(identities, dict):
            return out

        for identity, payload in identities.items():
            if not isinstance(identity, str) or not identity:
                continue
            if not isinstance(payload, dict):
                continue
            emb_raw = payload.get("embedding")
            try:
                emb = np.asarray(emb_raw, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if emb.size == 0:
                continue
            norm = float(np.linalg.norm(emb))
            if norm <= 1e-8:
                continue
            out[identity] = emb / norm
    except Exception as e:
        logger.warning("Failed to load pose-signature gallery from %s: %s", path, e)

    return out


def match_pose_signature(embedding: np.ndarray, gallery: Dict[str, np.ndarray]) -> Tuple[Optional[str], float]:
    """Return best (identity, cosine_score)."""
    if embedding is None or not isinstance(gallery, dict) or len(gallery) == 0:
        return None, 0.0

    try:
        emb = np.asarray(embedding, dtype=np.float32).reshape(-1)
    except Exception:
        return None, 0.0
    if emb.size == 0:
        return None, 0.0

    n = float(np.linalg.norm(emb))
    if n <= 1e-8:
        return None, 0.0
    emb = emb / n

    best_identity: Optional[str] = None
    best_score = -1.0
    for identity, ref in gallery.items():
        try:
            score = float(np.dot(emb, ref))
        except Exception:
            continue
        if score > best_score:
            best_score = score
            best_identity = identity

    if best_identity is None:
        return None, 0.0
    return best_identity, float(best_score)


class PoseSignatureCollector:
    """Collect pose frames and build a normalized embedding.

    Designed for single-person crops (one target per sequence), with optional
    zone-based weighting and per-frame quality gates.
    """

    def __init__(
        self,
        *,
        zones: Optional[Dict[str, object]] = None,
        keypoint_conf_thr: float = 0.25,
        min_upper_kp: int = 3,
        min_lower_kp: int = 4,
        min_frames: int = 24,
        smooth_alpha: float = 0.65,
        max_interp_gap_frames: int = 2,
    ):
        self.zones = zones
        self.keypoint_conf_thr = float(keypoint_conf_thr)
        self.min_upper_kp = int(max(1, min_upper_kp))
        self.min_lower_kp = int(max(1, min_lower_kp))
        self.min_frames = int(max(1, min_frames))
        self.smooth_alpha = float(np.clip(smooth_alpha, 0.0, 1.0))
        self.max_interp_gap_frames = int(max(0, max_interp_gap_frames))
        self._vec_dim = len(SIGNATURE_KEYPOINTS) * 2

        self.frames_seen = 0
        self.frames_used = 0
        self.frames_dropped_zone = 0
        self.frames_dropped_quality = 0

        self._vectors: List[np.ndarray] = []
        self._weights: List[float] = []
        self._valid_rows: List[bool] = []

    @staticmethod
    def _count_visible(conf: np.ndarray, indices: Tuple[int, ...], thr: float) -> int:
        c = 0
        for idx in indices:
            if idx < conf.shape[0] and float(conf[idx]) >= thr:
                c += 1
        return c

    @staticmethod
    def _dist(xy: np.ndarray, i: int, j: int) -> float:
        if i >= xy.shape[0] or j >= xy.shape[0]:
            return 0.0
        dx = float(xy[i, 0]) - float(xy[j, 0])
        dy = float(xy[i, 1]) - float(xy[j, 1])
        return float((dx * dx + dy * dy) ** 0.5)

    def _select_person_index(self, pose_result, count: int) -> int:
        if count <= 1:
            return 0
        try:
            boxes_xyxy = pose_result.boxes.xyxy
            if boxes_xyxy is None:
                return 0
            if hasattr(boxes_xyxy, "cpu"):
                boxes_xyxy = boxes_xyxy.cpu().numpy()
            boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
            if boxes_xyxy.ndim != 2 or boxes_xyxy.shape[0] == 0:
                return 0
            areas = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (boxes_xyxy[:, 3] - boxes_xyxy[:, 1])
            idx = int(np.argmax(areas))
            return int(max(0, min(idx, count - 1)))
        except Exception:
            return 0

    def _frame_vector(self, xy: np.ndarray, conf: np.ndarray) -> Tuple[Optional[np.ndarray], int, int]:
        upper_visible = self._count_visible(conf, UPPER_KEYPOINTS, self.keypoint_conf_thr)
        lower_visible = self._count_visible(conf, LOWER_KEYPOINTS, self.keypoint_conf_thr)
        if upper_visible < self.min_upper_kp or lower_visible < self.min_lower_kp:
            return None, upper_visible, lower_visible

        shoulder_mid = None
        hip_mid = None
        if conf.shape[0] > 6 and float(conf[5]) >= self.keypoint_conf_thr and float(conf[6]) >= self.keypoint_conf_thr:
            shoulder_mid = 0.5 * (xy[5] + xy[6])
        if conf.shape[0] > 12 and float(conf[11]) >= self.keypoint_conf_thr and float(conf[12]) >= self.keypoint_conf_thr:
            hip_mid = 0.5 * (xy[11] + xy[12])

        if hip_mid is not None:
            root = hip_mid
        elif shoulder_mid is not None:
            root = shoulder_mid
        else:
            valid_pts = []
            for idx in SIGNATURE_KEYPOINTS:
                if idx < conf.shape[0] and float(conf[idx]) >= self.keypoint_conf_thr:
                    valid_pts.append(xy[idx])
            if not valid_pts:
                return None, upper_visible, lower_visible
            root = np.mean(np.asarray(valid_pts, dtype=np.float32), axis=0)

        scale_candidates: List[float] = []
        d_shoulder = self._dist(xy, 5, 6)
        d_hip = self._dist(xy, 11, 12)
        if d_shoulder > 0:
            scale_candidates.append(d_shoulder)
        if d_hip > 0:
            scale_candidates.append(d_hip)
        if shoulder_mid is not None and hip_mid is not None:
            d_torso = float(np.linalg.norm(shoulder_mid - hip_mid))
            if d_torso > 0:
                scale_candidates.append(d_torso)

        if not scale_candidates:
            return None, upper_visible, lower_visible
        scale = float(max(scale_candidates))
        if scale <= 1e-6:
            return None, upper_visible, lower_visible

        vec: List[float] = []
        for idx in SIGNATURE_KEYPOINTS:
            if idx < conf.shape[0] and float(conf[idx]) >= self.keypoint_conf_thr:
                nx = (float(xy[idx, 0]) - float(root[0])) / scale
                ny = (float(xy[idx, 1]) - float(root[1])) / scale
                vec.extend([nx, ny])
            else:
                vec.extend([np.nan, np.nan])

        return np.asarray(vec, dtype=np.float32), upper_visible, lower_visible

    @staticmethod
    def _fill_nan_1d(col: np.ndarray) -> np.ndarray:
        out = col.copy()
        nans = np.isnan(out)
        if not np.any(nans):
            return out
        if np.all(nans):
            out[:] = 0.0
            return out
        idx = np.arange(out.shape[0], dtype=np.float32)
        out[nans] = np.interp(idx[nans], idx[~nans], out[~nans])
        return out

    def _smooth_ema(self, col: np.ndarray) -> np.ndarray:
        if col.size == 0:
            return col
        out = col.copy()
        a = self.smooth_alpha
        for i in range(1, out.shape[0]):
            out[i] = (a * out[i]) + ((1.0 - a) * out[i - 1])
        return out

    @staticmethod
    def _interpolate_short_gaps(
        values: np.ndarray,
        weights: np.ndarray,
        valid_rows: np.ndarray,
        max_gap: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Interpolate only short invalid-frame gaps bracketed by valid frames."""
        if values.ndim != 2 or values.shape[0] == 0:
            return values, weights

        n = int(values.shape[0])
        out_rows: List[np.ndarray] = []
        out_weights: List[float] = []

        i = 0
        while i < n:
            if bool(valid_rows[i]):
                out_rows.append(values[i])
                out_weights.append(float(weights[i]))
                i += 1
                continue

            start = i
            while i < n and not bool(valid_rows[i]):
                i += 1
            end = i  # first valid after gap (or n)

            gap = end - start
            prev_idx = start - 1
            next_idx = end if end < n else -1

            if (
                gap <= int(max_gap)
                and prev_idx >= 0
                and next_idx >= 0
                and bool(valid_rows[prev_idx])
                and bool(valid_rows[next_idx])
            ):
                prev_row = values[prev_idx]
                next_row = values[next_idx]
                prev_w = float(weights[prev_idx])
                next_w = float(weights[next_idx])
                for k in range(1, gap + 1):
                    alpha = float(k) / float(gap + 1)
                    row = prev_row + alpha * (next_row - prev_row)
                    w = (1.0 - alpha) * prev_w + alpha * next_w
                    out_rows.append(row.astype(np.float32))
                    out_weights.append(float(w))

        if not out_rows:
            return np.empty((0, values.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

        out_mat = np.stack(out_rows, axis=0).astype(np.float32)
        out_w = np.asarray(out_weights, dtype=np.float32)
        return out_mat, out_w

    def consume(self, pose_result, bbox: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Consume one pose inference result frame."""
        try:
            keypoints = getattr(pose_result, "keypoints", None)
            if keypoints is None:
                return
            xy = getattr(keypoints, "xy", None)
            if xy is None:
                return
            if hasattr(xy, "cpu"):
                xy = xy.cpu().numpy()
            xy = np.asarray(xy, dtype=np.float32)
            if xy.ndim != 3 or xy.shape[0] <= 0:
                return

            self.frames_seen += 1

            person_idx = self._select_person_index(pose_result, int(xy.shape[0]))
            pose_xy = xy[person_idx]

            conf = getattr(keypoints, "conf", None)
            if conf is None:
                pose_conf = np.ones((pose_xy.shape[0],), dtype=np.float32)
            else:
                if hasattr(conf, "cpu"):
                    conf = conf.cpu().numpy()
                conf = np.asarray(conf, dtype=np.float32)
                if conf.ndim == 2 and conf.shape[0] > person_idx:
                    pose_conf = conf[person_idx]
                else:
                    pose_conf = np.ones((pose_xy.shape[0],), dtype=np.float32)

            weight = zone_weight_for_bbox(bbox, self.zones)
            if weight <= 0.0:
                self.frames_dropped_zone += 1
                return

            vec, _upper, _lower = self._frame_vector(pose_xy, pose_conf)
            if vec is None:
                self.frames_dropped_quality += 1
                # Keep a timeline placeholder so short bad stretches can be
                # interpolated in finalize() when bracketed by good frames.
                self._vectors.append(np.full((self._vec_dim,), np.nan, dtype=np.float32))
                self._weights.append(float(weight))
                self._valid_rows.append(False)
                return

            self._vectors.append(vec)
            self._weights.append(float(weight))
            self._valid_rows.append(True)
            self.frames_used += 1
        except Exception:
            return

    def finalize(self) -> Dict[str, object]:
        """Return embedding and quality stats. Embedding may be None."""
        result: Dict[str, object] = {
            "embedding": None,
            "frames_seen": int(self.frames_seen),
            "frames_used": int(self.frames_used),
            "frames_dropped_zone": int(self.frames_dropped_zone),
            "frames_dropped_quality": int(self.frames_dropped_quality),
            "sequence_quality": 0.0,
        }

        if self.frames_seen > 0:
            result["sequence_quality"] = float(self.frames_used) / float(self.frames_seen)

        if self.frames_used < self.min_frames or len(self._vectors) == 0:
            return result

        mat_all = np.stack(self._vectors, axis=0).astype(np.float32)
        weights_all = np.asarray(self._weights, dtype=np.float32)
        valid_rows = np.asarray(self._valid_rows, dtype=bool)

        mat, weights = self._interpolate_short_gaps(
            mat_all,
            weights_all,
            valid_rows,
            max_gap=self.max_interp_gap_frames,
        )
        if mat.shape[0] <= 0:
            return result

        for d in range(mat.shape[1]):
            col = self._fill_nan_1d(mat[:, d])
            col = self._smooth_ema(col)
            mat[:, d] = col

        if weights.shape[0] != mat.shape[0] or float(np.sum(weights)) <= 1e-8:
            weights = np.ones((mat.shape[0],), dtype=np.float32)
        weights = weights / float(np.sum(weights))

        mean = np.sum(mat * weights[:, None], axis=0)
        var = np.sum(((mat - mean) ** 2) * weights[:, None], axis=0)
        std = np.sqrt(np.maximum(var, 0.0))

        if mat.shape[0] > 1:
            diffs = np.abs(np.diff(mat, axis=0))
            w2 = 0.5 * (weights[1:] + weights[:-1])
            if float(np.sum(w2)) <= 1e-8:
                w2 = np.ones((diffs.shape[0],), dtype=np.float32)
            w2 = w2 / float(np.sum(w2))
            dyn = np.sum(diffs * w2[:, None], axis=0)
        else:
            dyn = np.zeros((mat.shape[1],), dtype=np.float32)

        emb = np.concatenate([mean, std, dyn], axis=0).astype(np.float32)
        norm = float(np.linalg.norm(emb))
        if norm <= 1e-8:
            return result
        emb = emb / norm
        result["embedding"] = emb
        return result
