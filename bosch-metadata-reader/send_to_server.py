"""
send_to_server.py
Wire format (one message per frame document):
  {
    "type": "raw_vehicle",
    "doc": { ... same shape as send_to_api used to POST ... }
  }

Server side: expose a WebSocket endpoint that reads these messages and feeds
each `doc` into the same buffering path that /raw-vehicles uses today.
"""

import asyncio
import datetime
import json
import os
import queue
import threading
import time

import numpy as np
import pandas as pd
import websockets


# -------------------------------------------------------
# Config
# -------------------------------------------------------

# Default points at localhost; override with env var if running on another host
SERVER_WS_URL = os.environ.get(
    "SERVER_WS_URL",
    "ws://127.0.0.1:8000/ingest"
)

# Bounded queue so a stalled server can't blow up memory. When the queue is
# full we drop the OLDEST item — fresher frames are more valuable than stale
# ones for incident detection.
MAX_QUEUE_SIZE = 5000

# Reconnect backoff in seconds
RECONNECT_BACKOFF_MIN = 0.5
RECONNECT_BACKOFF_MAX = 10.0


# -------------------------------------------------------
# JSON sanitizing (same logic as the old send_to_api)
# -------------------------------------------------------

def _json_safe(obj):
    """Convert datetimes / numpy / pandas types into vanilla JSON-friendly values."""
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return pd.to_datetime(obj).isoformat()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# -------------------------------------------------------
# Background worker (one per process)
# -------------------------------------------------------

_OUTBOX: "queue.Queue[str]" = queue.Queue(maxsize=MAX_QUEUE_SIZE)
_WORKER_STARTED = False
_WORKER_LOCK = threading.Lock()


def _enqueue(payload_str: str) -> None:
    """Push a serialized message onto the outbox, dropping oldest on overflow."""
    try:
        _OUTBOX.put_nowait(payload_str)
    except queue.Full:
        # Drop the oldest item to make room — keep the stream fresh
        try:
            _OUTBOX.get_nowait()
        except queue.Empty:
            pass
        try:
            _OUTBOX.put_nowait(payload_str)
        except queue.Full:
            pass  # give up; queue is being hammered


async def _ws_sender_loop():
    """
    Owns the WebSocket. Reconnects forever. Drains _OUTBOX into the socket.
    Uses asyncio.to_thread to wait on the (synchronous) queue without blocking
    the event loop.
    """
    backoff = RECONNECT_BACKOFF_MIN

    while True:
        try:
            print(f"[send_to_server] Connecting to {SERVER_WS_URL} ...")
            async with websockets.connect(
                SERVER_WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=8 * 1024 * 1024,  # 8 MB frames; mapPath can get chunky
            ) as ws:
                print(f"[send_to_server] Connected. Backlog={_OUTBOX.qsize()}")
                backoff = RECONNECT_BACKOFF_MIN

                while True:
                    # Block (in a worker thread) until something's in the queue
                    payload_str = await asyncio.to_thread(_OUTBOX.get)
                    try:
                        await ws.send(payload_str)
                    except Exception as e:
                        # Connection died mid-send. Re-enqueue and reconnect.
                        print(f"[send_to_server] Send failed, requeueing: {e}")
                        _enqueue(payload_str)
                        raise

        except Exception as e:
            print(f"[send_to_server] WS error: {e}. Reconnecting in {backoff:.1f}s...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, RECONNECT_BACKOFF_MAX)


def _worker_thread_main():
    """Entry point for the dedicated sender thread. Runs the asyncio loop forever."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(_ws_sender_loop())
    finally:
        loop.close()


def _ensure_worker_started():
    """Lazily spin up the background sender thread on the first send call."""
    global _WORKER_STARTED
    if _WORKER_STARTED:
        return
    with _WORKER_LOCK:
        if _WORKER_STARTED:
            return
        t = threading.Thread(target=_worker_thread_main, name="send_to_server", daemon=True)
        t.start()
        _WORKER_STARTED = True
        print("[send_to_server] Background sender thread started")


# -------------------------------------------------------
# Public API (drop-in replacement for send_to_api)
# -------------------------------------------------------

def send_to_api(roadObjectData, *_, **__):
    """
    Drop-in replacement for the old send_to_api function.

    Same signature, same callers (collectData.pushObjectData) — but instead
    of doing an HTTP POST per frame, this serializes the document and pushes
    it into a queue drained by a single persistent WebSocket connection.

    Non-blocking: returns immediately after enqueueing.
    """
    try:
        _ensure_worker_started()

        safe_doc = _json_safe(roadObjectData)
        message = {"type": "raw_vehicle", "doc": safe_doc}
        payload_str = json.dumps(message)
        _enqueue(payload_str)

    except Exception as e:
        # Match the old function's behavior: never raise into the parser
        print(f"[send_to_server] Failed to enqueue: {e}")


# Optional: a graceful flush you can call before process exit
def flush(timeout: float = 5.0) -> int:
    """
    Wait up to `timeout` seconds for the queue to drain.
    Returns the number of items still queued when timeout expires.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _OUTBOX.empty():
            return 0
        time.sleep(0.05)
    return _OUTBOX.qsize()