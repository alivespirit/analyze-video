"""
Dispatch module for remote motion detection.

Provides:
- detect_motion_remote_async(): async remote dispatch (no executor needed)
- detect_motion_local(): lazy-loaded local fallback (avoids loading YOLO model at import time)
- worker_available(): cached health check

The async remote path allows multiple videos to be dispatched to the worker
concurrently (utilizing its 2+ slots), while local fallback uses the
single-worker motion_executor to avoid overloading the master CPU.
"""

import logging
import os
import re
import time

import httpx

logger = logging.getLogger()

# --- Configuration (master .env) ---
WORKER_URL = os.getenv("WORKER_URL", "http://10.0.0.2:8741")
WORKER_ENABLED = os.getenv("WORKER_ENABLED", "false").lower() == "true"
WORKER_TIMEOUT = float(os.getenv("WORKER_TIMEOUT", "120"))
WORKER_HEALTH_CACHE_SECONDS = float(os.getenv("WORKER_HEALTH_CACHE_SECONDS", "30"))
WORKER_MIN_BATTERY = int(os.getenv("WORKER_MIN_BATTERY", "5"))

# Wake-on-LAN configuration
WORKER_WAKE_ON_LAN = os.getenv("WORKER_WAKE_ON_LAN", "false").lower() in ("true", "1", "yes")
WORKER_WAKE_ON_LAN_MAC = os.getenv("WORKER_WAKE_ON_LAN_MAC", "")
WORKER_WAKE_ON_LAN_IFACE_IP = os.getenv("WORKER_WAKE_ON_LAN_IFACE_IP", "10.0.0.1")
_WOL_COOLDOWN_SECONDS = 300  # 5 minutes between WOL attempts
_last_wol_ts = 0.0

# --- Cached health state ---
_last_health_time = 0.0
_last_health_ok = False
_last_worker_battery = None  # last-seen worker battery percent

# --- Lazy-loaded local detect_motion ---
_detect_motion_fn = None


def detect_motion_local(input_video_path, output_dir, fast_processing=False):
    """Lazy-loading wrapper for detect_motion.

    Defers importing detect_motion (and loading the YOLO model) until
    the first time local processing is actually needed.
    """
    global _detect_motion_fn
    if _detect_motion_fn is None:
        logger.info("Loading local detect_motion (first local fallback)...")
        from detect_motion import detect_motion
        _detect_motion_fn = detect_motion
    return _detect_motion_fn(input_video_path, output_dir, fast_processing)


def get_worker_battery():
    """Return the last-known worker battery percent, or None if unknown."""
    return _last_worker_battery


def _send_wol_packet(mac_address: str):
    """Send Wake-on-LAN magic packet via the configured interface."""
    import socket
    mac_bytes = bytes.fromhex(mac_address.replace(":", "").replace("-", ""))
    magic = b'\xff' * 6 + mac_bytes * 16
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.bind((WORKER_WAKE_ON_LAN_IFACE_IP, 0))
        sock.sendto(magic, ("255.255.255.255", 9))
    finally:
        sock.close()


def _try_wol_if_needed():
    """Send WOL if enabled, master is plugged in, and cooldown has elapsed."""
    global _last_wol_ts
    if not WORKER_WAKE_ON_LAN or not WORKER_WAKE_ON_LAN_MAC:
        return
    now = time.monotonic()
    if now - _last_wol_ts < _WOL_COOLDOWN_SECONDS:
        return
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery and battery.power_plugged:
            _send_wol_packet(WORKER_WAKE_ON_LAN_MAC)
            _last_wol_ts = now
            logger.info("Sent WOL packet to %s (master plugged in)", WORKER_WAKE_ON_LAN_MAC)
        else:
            logger.debug("Skipping WOL: master not plugged in")
    except Exception as e:
        logger.warning("WOL failed: %s", e)


def _check_worker_health() -> bool:
    """Query GET /health with a short timeout. Returns True if worker is available.

    On failure, retries once after a short delay before returning False, to avoid
    marking the worker unavailable on a transient network blip.
    """
    global _last_worker_battery
    for attempt in range(2):
        try:
            resp = httpx.get(f"{WORKER_URL}/health", timeout=2.0)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") != "ok":
                return False
            battery = data.get("battery_percent")
            _last_worker_battery = battery
            if battery is not None and battery < WORKER_MIN_BATTERY:
                logger.warning("Worker battery low (%s%%), skipping remote dispatch.", battery)
                return False
            return True
        except Exception as e:
            if attempt == 0:
                logger.warning("Worker health check failed: %r. Retrying...", e)
                time.sleep(1.0)
            else:
                logger.warning("Worker health check failed: %r", e)
                _try_wol_if_needed()
                return False


def worker_available() -> bool:
    """Cached health check. Re-checks at most every WORKER_HEALTH_CACHE_SECONDS."""
    global _last_health_time, _last_health_ok
    now = time.monotonic()
    if now - _last_health_time < WORKER_HEALTH_CACHE_SECONDS:
        return _last_health_ok
    was_ok = _last_health_ok
    _last_health_ok = _check_worker_health()
    _last_health_time = now
    if was_ok and not _last_health_ok:
        logger.warning("Worker became unavailable.")
    elif not was_ok and _last_health_ok:
        logger.info("Worker is back online.")
    return _last_health_ok


def invalidate_worker_health():
    """Reset health cache so the next call re-checks immediately."""
    global _last_health_time
    _last_health_time = 0.0


_WORKER_TAG_RE = re.compile(
    r"^(?P<head>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} - [A-Z]+ -)(?: (?P<bracket>\[[^\]]+\]))? (?P<tail>.*)$"
)


def _replay_worker_logs(logs):
    """Replay pre-formatted log lines from worker directly into master's log handlers.

    Inserts [W] after the [filename] bracket (or after the level if no bracket) to keep lines sortable by timestamp.
    Bypasses the master's formatter to preserve the worker's original timestamps and levels.
    """
    if not logs:
        return
    for line in logs:
        m = _WORKER_TAG_RE.match(line)
        if m:
            bracket = m.group('bracket')
            if bracket:
                prefixed = f"{m.group('head')} {bracket} [W] {m.group('tail')}\n"
            else:
                prefixed = f"{m.group('head')} [W] {m.group('tail')}\n"
        else:
            prefixed = f"[W] {line}\n"
        for handler in logger.handlers:
            try:
                if hasattr(handler, "stream"):
                    handler.stream.write(prefixed)
                    handler.stream.flush()
                else:
                    logger.info("[W] %s", line)
            except Exception:
                pass


async def detect_motion_remote_async(file_path, output_dir, fast_processing=False):
    """
    Async dispatch of detect_motion to the remote worker via HTTP.

    Sends master-perspective paths; worker handles translation, local copy,
    processing, and copying results back to CIFS mount.

    On ReadError (connection dropped mid-response) retries once — the worker
    processes a video in ~20 s so a duplicate run is cheaper than local fallback.

    Returns the same dict as detect_motion() with master-perspective paths.
    Raises on any failure (caller should catch and fall back to local).
    """
    payload = {
        "video_path": file_path,
        "output_dir": output_dir,
        "fast_processing": fast_processing,
    }
    file_basename = os.path.basename(file_path)
    for attempt in range(2):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{WORKER_URL}/detect-motion",
                    json=payload,
                    timeout=WORKER_TIMEOUT,
                )
            resp.raise_for_status()
            data = resp.json()
            _replay_worker_logs(data.get("logs"))
            return data["result"]
        except httpx.ReadError as e:
            if attempt == 0:
                logger.warning("[%s] Worker ReadError on attempt 1, retrying: %r", file_basename, e)
            else:
                raise
