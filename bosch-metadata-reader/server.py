from fastapi import FastAPI, BackgroundTasks
from datetime import datetime

# =========================
# Local imports (PURE)
# =========================

from combined_statistics import run_batches, load_all_combined_stats
# from ffmpegreader import start_ingest

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
    """
    Sanity check endpoint.
    Nothing should run when this is hit.
    """
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat()
    }

# =========================
# Stats pipeline (manual trigger)
# =========================

@app.post("/stats/run")
def run_combined_stats(
    background_tasks: BackgroundTasks,
    batch_size: int = 100_000
):
    """
    Manually triggers combined_statistics batch processing.
    Raw → combined_stats.
    """
    background_tasks.add_task(run_batches, batch_size)

    return {
        "status": "accepted",
        "message": "Combined stats job started",
        "batch_size": batch_size
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
    # 1) Load stats from Mongo
    df = load_all_combined_stats(limit=limit)

    if df.empty:
        return {
            "rows_analyzed": 0,
            "count": 0,
            "incidents": []
        }

    # 2) Apply ML preprocessing (scaling)
    df, _ = scale_per_location(df)

    # 3) Load models
    models = load_latest_models()

    # 4) Run detection
    incidents = detect_incidents(df, models)

    return {
        "rows_analyzed": len(df),
        "count": len(incidents),
        "incidents": incidents
    }

# =========================
# Ingestion
# =========================

@app.post("/ingest/start")
def start_camera_ingest(
    camera: str,
    xml_path: str,
    background: BackgroundTasks
):
    """
    Starts ffmpeg XML ingestion in the background.
    NOTHING runs unless this endpoint is called.
    """
    background.add_task(start_ingest, camera, xml_path)

    return {
        "status": "ingestion started",
        "camera": camera,
        "xml_path": xml_path
    }
