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

import asyncio
import logging
import os
import re
import socket
import time

import httpx

logger = logging.getLogger()

# --- Configuration (master .env) ---
WORKER_URL = os.getenv("WORKER_URL", "http://10.0.0.2:8741")
WORKER_ENABLED = os.getenv("WORKER_ENABLED", "false").lower() == "true"
WORKER_TIMEOUT = float(os.getenv("WORKER_TIMEOUT", "120"))
# Dispatch retry policy. The direct master<->worker link can briefly flap (NIC
# EEE/power-save renegotiation) and drop the idle HTTP connection while the worker
# processes. Retry a few times with a backoff long enough for the link to re-establish,
# rather than immediately falling back to the (much slower) local path.
WORKER_DISPATCH_ATTEMPTS = int(os.getenv("WORKER_DISPATCH_ATTEMPTS", "3"))
WORKER_DISPATCH_BACKOFF = float(os.getenv("WORKER_DISPATCH_BACKOFF", "10.0"))
WORKER_TCP_KEEPALIVE_IDLE = int(os.getenv("WORKER_TCP_KEEPALIVE_IDLE", "15"))
# Cap the connect phase so a retry against a still-down link fails fast instead of
# hanging for the full WORKER_TIMEOUT (which is meant for the read/processing wait).
WORKER_CONNECT_TIMEOUT = float(os.getenv("WORKER_CONNECT_TIMEOUT", "10.0"))
WORKER_HEALTH_CACHE_SECONDS = float(os.getenv("WORKER_HEALTH_CACHE_SECONDS", "30"))
WORKER_MIN_BATTERY = int(os.getenv("WORKER_MIN_BATTERY", "5"))

# Wake-on-LAN configuration
WORKER_WAKE_ON_LAN = os.getenv("WORKER_WAKE_ON_LAN", "false").lower() in ("true", "1", "yes")
WORKER_WAKE_ON_LAN_MAC = os.getenv("WORKER_WAKE_ON_LAN_MAC", "")
WORKER_WAKE_ON_LAN_IFACE_IP = os.getenv("WORKER_WAKE_ON_LAN_IFACE_IP", "10.0.0.1")
WORKER_WAKE_ON_LAN_BROADCAST_IP = os.getenv("WORKER_WAKE_ON_LAN_BROADCAST_IP", "10.0.0.255")
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
        sock.sendto(magic, (WORKER_WAKE_ON_LAN_BROADCAST_IP, 9))
    finally:
        sock.close()


def try_wol_if_needed():
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
                _last_worker_battery = None
                try_wol_if_needed()
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
        logger.info("Worker is online.")
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


def _keepalive_socket_options():
    """TCP keep-alive tuning for the dispatch connection. The socket sits idle while
    the worker processes (video/artifacts move over SMB, not this socket), so keep-alive
    probes keep the link warm — defeating NIC idle power-down — and surface a dropped
    peer quickly. The TCP_KEEP* constants vary by platform/Python, hence the hasattr guards."""
    opts = [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
    if hasattr(socket, "TCP_KEEPIDLE"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, WORKER_TCP_KEEPALIVE_IDLE))
    if hasattr(socket, "TCP_KEEPINTVL"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5))
    if hasattr(socket, "TCP_KEEPCNT"):
        opts.append((socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3))
    return opts


def _new_worker_client():
    """AsyncClient with TCP keep-alive enabled. Falls back to a default client if the
    installed httpx predates `socket_options` support (added in httpx 0.26)."""
    try:
        transport = httpx.AsyncHTTPTransport(socket_options=_keepalive_socket_options())
        return httpx.AsyncClient(transport=transport)
    except TypeError:
        return httpx.AsyncClient()


async def detect_motion_remote_async(file_path, output_dir, fast_processing=False):
    """
    Async dispatch of detect_motion to the remote worker via HTTP.

    Sends master-perspective paths; worker handles translation, local copy,
    processing, and copying results back to CIFS mount.

    Transient transport errors (a link flap dropping the idle connection, connect
    failures) are retried up to WORKER_DISPATCH_ATTEMPTS times with a
    WORKER_DISPATCH_BACKOFF pause between tries — long enough for a flapped NIC link
    to re-establish. The connection uses TCP keep-alive to stay warm during the
    worker's idle processing window. A duplicate worker run is cheaper than the local
    fallback. HTTP status errors and read timeouts are not retried (real failures /
    genuinely-slow processing → fall back to local).

    Returns the same dict as detect_motion() with master-perspective paths.
    Raises on any failure (caller should catch and fall back to local).
    """
    payload = {
        "video_path": file_path,
        "output_dir": output_dir,
        "fast_processing": fast_processing,
    }
    file_basename = os.path.basename(file_path)
    # Retry only transport-level failures (link flap / connect drop). NOT ReadTimeout
    # (worker genuinely too slow) or HTTPStatusError (real processing error) — those
    # propagate so the caller falls back to local.
    retryable = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadError,
        httpx.WriteError,
        httpx.RemoteProtocolError,
    )
    for attempt in range(WORKER_DISPATCH_ATTEMPTS):
        try:
            async with _new_worker_client() as client:
                resp = await client.post(
                    f"{WORKER_URL}/detect-motion",
                    json=payload,
                    timeout=httpx.Timeout(WORKER_TIMEOUT, connect=WORKER_CONNECT_TIMEOUT),
                )
            resp.raise_for_status()
            data = resp.json()
            _replay_worker_logs(data.get("logs"))
            return data["result"]
        except retryable as e:
            if attempt < WORKER_DISPATCH_ATTEMPTS - 1:
                logger.warning(
                    "[%s] Worker transport error on attempt %d/%d, retrying in %.0fs: %r",
                    file_basename, attempt + 1, WORKER_DISPATCH_ATTEMPTS,
                    WORKER_DISPATCH_BACKOFF, e,
                )
                await asyncio.sleep(WORKER_DISPATCH_BACKOFF)
            else:
                logger.warning(
                    "[%s] Worker dispatch failed after %d attempts: %r",
                    file_basename, WORKER_DISPATCH_ATTEMPTS, e,
                )
                raise
