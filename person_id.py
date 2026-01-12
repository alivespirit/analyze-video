import os
import hashlib
import logging
import cv2
import numpy as np
from openvino import Core


logger = logging.getLogger()

# Cache gallery embeddings per path to avoid reloading within the same process
_GALLERY_CACHE: dict[str, list[np.ndarray]] = {}
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "temp")


class PersonReID:
    """OpenVINO-based person re-identification helper.

    Loads `person-reidentification-retail-0288` and builds a reference gallery
    from images in a folder. Provides `identify()` to compare a person crop
    against the gallery using cosine similarity of normalized embeddings.
    """

    def __init__(self, model_path: str, gallery_path: str, threshold: float = 0.65, file_basename: str | None = None):
        self.threshold = float(threshold)
        self.gallery_vectors: list[np.ndarray] = []
        self.file_basename = file_basename

        def _lp():
            return f"[{self.file_basename}] " if self.file_basename else ""
        self._lp = _lp

        logger.debug(f"{self._lp()}ReID: Initializing model from {model_path}")
        self.core = Core()
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.output_layer = self.compiled_model.output(0)

        self.load_gallery(gallery_path)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        # Model expects 128x256 resolution, BGR input
        img = cv2.resize(img, (128, 256))
        # Change Layout from HWC to NCHW
        img = img.transpose((2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, 0)
        return img.astype(np.float32)

    def get_embedding(self, img_crop: np.ndarray) -> np.ndarray:
        input_data = self.preprocess(img_crop)
        # Run inference
        result = self.compiled_model([input_data])[self.output_layer]
        # Normalize the vector (crucial for cosine similarity)
        vec = result[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def load_gallery(self, gallery_path: str) -> None:
        if not os.path.exists(gallery_path):
            logger.warning(f"{self._lp()}ReID: Gallery not found at {gallery_path}.")
            return

        # Use cached embeddings if available
        cached = _GALLERY_CACHE.get(gallery_path)
        if cached is not None:
            self.gallery_vectors = cached
            logger.debug(f"{self._lp()}ReID: Using cached gallery with {len(cached)} vector(s) from {gallery_path}.")
            return

        # Prepare disk cache paths (per-gallery, per-absolute path)
        try:
            os.makedirs(_CACHE_DIR, exist_ok=True)
        except Exception:
            pass
        gallery_key = hashlib.sha1(os.path.abspath(gallery_path).encode("utf-8")).hexdigest()[:16]
        cache_npz = os.path.join(_CACHE_DIR, f"reid_gallery_cache_{gallery_key}.npz")

        # Scan gallery files and compute freshness indicators
        files = [f for f in os.listdir(gallery_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        current_count = len(files)
        latest_mtime = 0.0
        for f in files:
            try:
                latest_mtime = max(latest_mtime, os.path.getmtime(os.path.join(gallery_path, f)))
            except Exception:
                continue

        # Try loading disk cache if up-to-date and counts match
        if os.path.exists(cache_npz):
            try:
                npz_mtime = os.path.getmtime(cache_npz)
                with np.load(cache_npz, allow_pickle=False) as data:
                    vectors = data["vectors"] if "vectors" in data.files else None
                    saved_count = int(data["count"]) if "count" in data.files else (vectors.shape[0] if vectors is not None else 0)
                if vectors is not None and npz_mtime >= latest_mtime and saved_count == current_count:
                    # Convert to list of 1D arrays
                    self.gallery_vectors = [vectors[i] for i in range(vectors.shape[0])]
                    _GALLERY_CACHE[gallery_path] = self.gallery_vectors
                    logger.debug(f"{self._lp()}ReID: Loaded {saved_count} vector(s) from disk cache {cache_npz}.")
                    return
                else:
                    logger.debug(f"{self._lp()}ReID: Disk cache outdated or count mismatch (cache_mtime={npz_mtime:.0f}, latest_mtime={latest_mtime:.0f}, count={current_count}). Recomputing.")
            except Exception as e:
                logger.debug(f"{self._lp()}ReID: Failed to use disk cache {cache_npz}: {e}. Recomputing.")

        logger.debug(f"{self._lp()}ReID: Loading reference gallery from {gallery_path}...")
        count = 0
        for filename in os.listdir(gallery_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(gallery_path, filename)
            img = cv2.imread(path)
            if img is None:
                logger.warning(f"{self._lp()}ReID: Failed to read gallery image: {path}")
                continue
            try:
                vec = self.get_embedding(img)
                self.gallery_vectors.append(vec)
                count += 1
            except Exception as e:
                logger.warning(f"{self._lp()}ReID: Embedding failed for {path}: {e}")
        logger.debug(f"{self._lp()}ReID: Loaded {count} reference image(s).")
        _GALLERY_CACHE[gallery_path] = self.gallery_vectors

        # Persist to disk cache for reuse across worker processes
        if count > 0:
            try:
                vectors_array = np.stack(self.gallery_vectors)
                np.savez(cache_npz, vectors=vectors_array, count=count)
                logger.debug(f"{self._lp()}ReID: Saved {count} vector(s) to disk cache {cache_npz}.")
            except Exception as e:
                logger.debug(f"{self._lp()}ReID: Failed to save disk cache {cache_npz}: {e}")

    def identify(self, person_crop: np.ndarray) -> tuple[bool, float]:
        """Return (is_match, max_score) against loaded gallery."""
        if len(self.gallery_vectors) == 0:
            return False, 0.0

        target_vec = self.get_embedding(person_crop)
        max_score = 0.0
        # Compare against every photo in gallery using cosine similarity
        for ref_vec in self.gallery_vectors:
            score = float(np.dot(target_vec, ref_vec))
            if score > max_score:
                max_score = score

        is_match = max_score >= self.threshold
        return is_match, max_score
