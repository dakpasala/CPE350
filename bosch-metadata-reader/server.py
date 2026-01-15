from fastapi import FastAPI
from datetime import datetime
from typing import List, Dict

# =========================
# Local imports
# =========================

from combined_statistics import process_raw_docs, save_stats, load_all_combined_stats

from incident_detection.engine import detect_incidents
from incident_detection.models import load_latest_models
from incident_detection.data import scale_per_location

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
    Runs feature extraction immediately.
    Writes ONLY combined_stats to Mongo.
    """
    df = process_raw_docs(payload)

    if not df.empty:
        save_stats(df)

    return {
        "received": len(payload),
        "processed_rows": len(df)
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
