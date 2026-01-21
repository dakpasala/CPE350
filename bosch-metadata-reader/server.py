from fastapi import FastAPI
from datetime import datetime, timedelta
from typing import List, Dict

from collections import defaultdict, deque
import pandas as pd


# =========================
# Local imports
# =========================

from combined_statistics import (
    process_raw_docs,
    save_stats,
    load_all_combined_stats,
)

from incident_detection.engine import detect_incidents
from incident_detection.models import load_latest_models
from incident_detection.data import scale_per_location


# =========================
# Buffering config
# =========================

BUFFER_SECONDS = 15        # ⭐ CHANGED (was 5 — too short for trajectories)
MIN_BUFFER_DOCS = 3        # ⭐ NEW: minimum docs before feature extraction

# location -> deque of raw vehicle dicts
RAW_BUFFER = defaultdict(deque)

# location -> last processed timestamp
LAST_PROCESSED_TS = {}     # ⭐ NEW


# =========================
# App init
# =========================

app = FastAPI(
    title="Traffic Incident Detection API",
    version="1.0"
)


# =========================
# Health
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat()
    }


# =========================
# 🔥 RAW INGESTION (ffmpegreader hits this)
# =========================

@app.post("/raw-vehicles")
def ingest_raw_vehicles(payload: List[Dict]):
    """
    Receives raw vehicle docs from ffmpegreader.
    Buffers livestream data and writes ONLY combined_stats to Mongo.
    """

    if not payload:
        return {"received": 0, "processed_rows": 0}

    processed_rows = 0

    # -------------------------
    # Buffer incoming docs
    # -------------------------
    for doc in payload:
        location = doc.get("location", "unknown")

        ts = doc.get("timestamp")
        if isinstance(ts, str):
            ts = pd.to_datetime(ts, errors="coerce")
        elif isinstance(ts, datetime):
            ts = ts
        else:
            ts = pd.NaT

        if pd.isna(ts):
            continue

        doc["_parsed_ts"] = ts
        RAW_BUFFER[location].append(doc)

    # -------------------------
    # Process buffered data
    # -------------------------
    for location, buf in RAW_BUFFER.items():
        if not buf:
            continue

        newest_ts = buf[-1]["_parsed_ts"]
        cutoff = newest_ts - timedelta(seconds=BUFFER_SECONDS)

        # Drop old data outside buffer window
        while buf and buf[0]["_parsed_ts"] < cutoff:
            buf.popleft()

        buffered_payload = list(buf)

        # ⭐ NEW: avoid reprocessing already-consumed data
        last_ts = LAST_PROCESSED_TS.get(location)
        if last_ts is not None:
            buffered_payload = [
                d for d in buffered_payload
                if d["_parsed_ts"] > last_ts
            ]

        # Not enough temporal support yet
        if len(buffered_payload) < MIN_BUFFER_DOCS:
            continue

        # -------------------------
        # Feature extraction
        # -------------------------
        df = process_raw_docs(buffered_payload)

        if df.empty:
            continue

        # ⭐ NEW: drop rows with no motion signal
        if "speed_mps" in df.columns:
            df = df[df["speed_mps"].notna()]

        if df.empty:
            continue

        # -------------------------
        # Persist derived stats
        # -------------------------
        save_stats(df)
        processed_rows += len(df)

        # ⭐ NEW: mark progress
        LAST_PROCESSED_TS[location] = buffered_payload[-1]["_parsed_ts"]

    return {
        "received": len(payload),
        "processed_rows": processed_rows,
        "buffer_seconds": BUFFER_SECONDS
    }


# =========================
# Incident detection
# =========================

@app.post("/incidents/run")
def run_incident_detection(limit: int = 10_000):
    """
    Runs incident detection on existing combined_stats.
    NO feature computation happens here.
    """

    df = load_all_combined_stats(limit=limit)

    if df.empty:
        return {
            "rows_analyzed": 0,
            "count": 0,
            "incidents": []
        }

    df, _ = scale_per_location(df)
    models = load_latest_models()
    incidents = detect_incidents(df, models)

    return {
        "rows_analyzed": len(df),
        "count": len(incidents),
        "incidents": incidents
    }
