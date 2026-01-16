from fastapi import FastAPI
from datetime import datetime
from typing import List, Dict

from collections import defaultdict, deque
from datetime import timedelta
import pandas as pd


# =========================
# Local imports
# =========================

from combined_statistics import process_raw_docs, save_stats, load_all_combined_stats

from incident_detection.engine import detect_incidents
from incident_detection.models import load_latest_models
from incident_detection.data import scale_per_location

BUFFER_SECONDS = 5

# location -> deque of raw vehicle dicts
RAW_BUFFER = defaultdict(deque)

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
    Writes ONLY combined_stats to Mongo.
    """

    if not payload:
        return {"received": 0, "processed_rows": 0}

    now = datetime.utcnow()
    processed_rows = 0

    for doc in payload:
        location = doc.get("location", "unknown")

        # Parse timestamp safely
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

    for location, buf in RAW_BUFFER.items():
        if not buf:
            continue

        newest_ts = buf[-1]["_parsed_ts"]
        cutoff = newest_ts - timedelta(seconds=BUFFER_SECONDS)

        # keep only last N seconds
        while buf and buf[0]["_parsed_ts"] < cutoff:
            buf.popleft()

        # Not enough temporal data yet
        if len(buf) < 2:
            continue

        buffered_payload = list(buf)
        df = process_raw_docs(buffered_payload)

        if not df.empty:
            save_stats(df)
            processed_rows += len(df)

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
