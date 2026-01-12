from fastapi import FastAPI, BackgroundTasks
from datetime import datetime, timedelta

# =========================
# Import your modules
# =========================

from ffmpegreader import start_ingest
from combined_statistics import run_feature_pipeline

from incident_detection.engine import detect_incidents
from incident_detection.models import load_latest_models
from incident_detection.data import load_data, scale_per_location

# =========================
# App init
# =========================

app = FastAPI(
    title="Traffic Incident Detection API",
    version="1.0"
)

@app.post("/incidents/from-csv")
def run_incidents_from_csv(
    csv_path: str = "/Users/dakshesh/CPE 350/bosch-metadata-reader/combined_vehicle_stats_expandedNEW.csv"

):
    """
    Debug / validation endpoint.
    Runs incident detection directly on a CSV.
    """

    # 1) Load CSV → DataFrame
    df = load_data(csv_path)
    df, _ = scale_per_location(df)

    # 2) Load trained models
    models = load_latest_models()

    # 3) Run detection
    incidents = detect_incidents(df, models)

    return {
        "csv_path": csv_path,
        "rows": len(df),
        "incident_count": len(incidents),
        "incidents": incidents,
    }

# =========================
# Health
# =========================

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# =========================
# Ingestion routes
# =========================

@app.post("/ingest/start")
def start_camera_ingest(
    camera: str,
    xml_path: str,
    background: BackgroundTasks
):
    """
    Starts ffmpeg XML ingestion in background.
    """
    background.add_task(start_ingest, camera, xml_path)
    return {
        "status": "ingestion started",
        "camera": camera,
        "xml_path": xml_path
    }

# =========================
# Feature routes
# =========================

@app.post("/features/run")
def run_features(limit: int | None = None):
    """
    Runs combined_statistics feature pipeline.
    Returns summary only.
    """
    df = run_feature_pipeline(limit=limit)
    return {
        "rows_processed": len(df),
        "columns": list(df.columns)
    }

# =========================
# Incident routes
# =========================

@app.post("/incidents/run")
def run_incident_detection(last_seconds: int = 30):
    """
    Runs incident detection on recent data.
    Returns detected incidents as JSON.
    """
    # ⛔ placeholder: you will swap this for a Mongo window query
    df = run_feature_pipeline()

    models = load_latest_models()
    incidents = detect_incidents(df, models)

    return {
        "count": len(incidents),
        "incidents": incidents
    }

# =========================
# Convenience route
# =========================

@app.get("/incidents/latest")
def get_latest_incidents():
    """
    Stub for Mongo-backed query.
    """
    return {
        "message": "Query latest incidents from Mongo (not wired yet)"
    }
