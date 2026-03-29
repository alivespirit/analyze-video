"""
Remote worker server for motion detection.

Runs on the worker machine (10.0.0.2) as a FastAPI app.
Receives master-perspective paths, translates them, copies the video locally,
runs detect_motion, copies results to CIFS mount, and returns the result dict.

Start with:
    uvicorn worker.server:app --host 0.0.0.0 --port 8741
"""

import asyncio
import concurrent.futures
import logging
import os
import shutil
import tempfile
import threading

import psutil
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()

# --- Path mapping configuration ---
MASTER_PATH_PREFIX = os.getenv("MASTER_PATH_PREFIX", "C:\\NAS\\")
WORKER_PATH_PREFIX = os.getenv("WORKER_PATH_PREFIX", "/mnt/nas/")

# Normalize master prefix for comparison (forward-slash, lowercase)
_MASTER_PREFIX_NORM = MASTER_PATH_PREFIX.replace("\\", "/").rstrip("/") + "/"
_MASTER_PREFIX_NORM_UPPER = _MASTER_PREFIX_NORM.upper()

# --- Worker settings ---
WORKER_MAX_CONCURRENT = int(os.getenv("WORKER_MAX_CONCURRENT", "2"))

# --- Logging setup ---
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)

# Suppress noisy third-party loggers
for name in ("httpx", "uvicorn.access", "moviepy", "ultralytics"):
    logging.getLogger(name).setLevel(logging.WARNING)

# --- Import detect_motion after env is loaded ---
from detect_motion import detect_motion  # noqa: E402

# --- Executor and task tracking ---
executor = concurrent.futures.ThreadPoolExecutor(max_workers=WORKER_MAX_CONCURRENT)
_active_tasks = 0
_task_lock = asyncio.Lock()

app = FastAPI(title="Analyze Video Worker")


# --- Path translation helpers ---

def master_to_worker_path(master_path: str) -> str:
    """Translate a master (Windows) path to a worker (Linux/CIFS) path.

    Example: C:\\NAS\\foo\\bar → /mnt/nas/foo/bar
    """
    p = master_path.replace("\\", "/")
    if p.upper().startswith(_MASTER_PREFIX_NORM_UPPER):
        p = WORKER_PATH_PREFIX.rstrip("/") + "/" + p[len(_MASTER_PREFIX_NORM):]
    return p


def worker_to_master_path(worker_path: str) -> str:
    """Translate a worker (Linux/CIFS) path back to a master (Windows) path.

    Example: /mnt/nas/foo/bar → C:\\NAS\\foo\\bar
    """
    prefix = WORKER_PATH_PREFIX.rstrip("/") + "/"
    if worker_path.startswith(prefix):
        remainder = worker_path[len(prefix):]
        return MASTER_PATH_PREFIX.rstrip("\\") + "\\" + remainder.replace("/", "\\")
    return worker_path


def translate_result_paths(result: dict, local_output_dir: str, cifs_output_dir: str) -> dict:
    """Rewrite local output paths in the result dict to master-perspective paths.

    Files have already been copied from local_output_dir to cifs_output_dir.
    We replace the local prefix with the CIFS prefix, then convert to master path.
    """
    local_prefix = local_output_dir.rstrip("/") + "/"
    cifs_prefix = cifs_output_dir.rstrip("/") + "/"

    def rewrite(path):
        if not path or not isinstance(path, str):
            return path
        if path.startswith(local_prefix):
            cifs_path = cifs_prefix + path[len(local_prefix):]
            return worker_to_master_path(cifs_path)
        return worker_to_master_path(path)

    if result.get("clip_path"):
        result["clip_path"] = rewrite(result["clip_path"])

    if result.get("insignificant_frames"):
        result["insignificant_frames"] = [rewrite(p) for p in result["insignificant_frames"]]

    if result.get("reid") and isinstance(result["reid"], dict):
        if result["reid"].get("best_path"):
            result["reid"]["best_path"] = rewrite(result["reid"]["best_path"])

    return result


# --- Log capture handler ---

class LogCaptureHandler(logging.Handler):
    """Temporary handler that captures formatted log lines from a specific thread only."""

    def __init__(self, formatter, thread_id):
        super().__init__()
        self.setFormatter(formatter)
        self.thread_id = thread_id
        self.records = []

    def emit(self, record):
        if record.thread != self.thread_id:
            return
        try:
            self.records.append(self.format(record))
        except Exception:
            pass


# --- Copy helpers ---

def copy_outputs(local_output_dir: str, cifs_output_dir: str) -> None:
    """Copy all generated output files from local temp to CIFS mount."""
    for root, dirs, files in os.walk(local_output_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, local_output_dir)
            dst = os.path.join(cifs_output_dir, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)


# --- Worker processing function ---

def process_video(video_path_cifs: str, cifs_output_dir: str, fast_processing: bool):
    """Copy video locally, run detect_motion, copy results to CIFS mount.

    Returns (result_dict, captured_log_lines).
    """
    local_tmp = tempfile.mkdtemp(prefix="worker_")
    local_video = os.path.join(local_tmp, os.path.basename(video_path_cifs))
    local_output = os.path.join(local_tmp, "output")
    os.makedirs(local_output, exist_ok=True)

    try:
        # Copy video from CIFS to local for fast I/O
        shutil.copy2(video_path_cifs, local_video)

        # Capture logs during detect_motion (filtered to this thread only)
        capture = LogCaptureHandler(log_formatter, threading.current_thread().ident)
        logger.addHandler(capture)
        try:
            result = detect_motion(local_video, local_output, fast_processing)
        finally:
            logger.removeHandler(capture)

        # Copy output files to CIFS mount
        copy_outputs(local_output, cifs_output_dir)

        # Translate paths in result to master perspective
        translate_result_paths(result, local_output, cifs_output_dir)

        return result, capture.records

    finally:
        # Clean up local temp
        shutil.rmtree(local_tmp, ignore_errors=True)


# --- Request model ---

class DetectMotionRequest(BaseModel):
    video_path: str
    output_dir: str
    fast_processing: bool = False


# --- Endpoints ---

@app.get("/health")
async def health():
    battery = psutil.sensors_battery()
    battery_pct = battery.percent if battery else None
    return {
        "status": "ok",
        "active_tasks": _active_tasks,
        "max_tasks": WORKER_MAX_CONCURRENT,
        "battery_percent": battery_pct,
    }


@app.post("/detect-motion")
async def detect_motion_endpoint(req: DetectMotionRequest):
    global _active_tasks

    # Translate master paths to worker CIFS paths
    video_path_cifs = master_to_worker_path(req.video_path)
    cifs_output_dir = master_to_worker_path(req.output_dir)

    video_basename = os.path.basename(video_path_cifs)
    logger.info("[%s] Received from master. fast_processing=%s", video_basename, req.fast_processing)

    async with _task_lock:
        _active_tasks += 1

    try:
        loop = asyncio.get_running_loop()
        result, logs = await loop.run_in_executor(
            executor, process_video, video_path_cifs, cifs_output_dir, req.fast_processing
        )
        logger.info("[%s] Processing complete. Status: %s", video_basename, result.get("status"))
        return {"result": result, "logs": logs}
    except Exception as e:
        logger.error("[%s] Processing failed: %s", video_basename, e, exc_info=True)
        return {"result": {"status": "error", "clip_path": None, "insignificant_frames": []}, "logs": [str(e)]}
    finally:
        async with _task_lock:
            _active_tasks -= 1
