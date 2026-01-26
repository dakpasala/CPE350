from fastapi import FastAPI
from datetime import datetime, timedelta
from typing import List, Dict

from collections import defaultdict, deque
import pandas as pd
from threading import Thread, Lock


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

WINDOW_SECONDS = 15        # Fixed window: accumulate for 15s, then process

# location -> deque of raw vehicle dicts
RAW_BUFFER = defaultdict(deque)

# location -> window start time
WINDOW_START = {}

# Thread safety
BUFFER_LOCK = Lock()


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
    with BUFFER_LOCK:
        buffer_stats = {}
        for loc in RAW_BUFFER.keys():
            window_start = WINDOW_START.get(loc)
            elapsed = (datetime.utcnow() - window_start).total_seconds() if window_start else 0
            buffer_stats[loc] = {
                "buffered": len(RAW_BUFFER[loc]),
                "window_elapsed": round(elapsed, 1),
                "window_closes_in": round(WINDOW_SECONDS - elapsed, 1)
            }
    
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "buffer_stats": buffer_stats,
        "window_seconds": WINDOW_SECONDS,
    }


# =========================
# Background processor
# =========================

def process_window_async(location: str, docs: list[Dict]):
    """
    Processes one complete 15-second window.
    Then automatically runs incident detection on the new data.
    """
    
    if not docs:
        return
    
    print(f"🔄 Processing {location} window | docs={len(docs)}")
    
    try:
        # Feature extraction on full window
        df = process_raw_docs(docs)
    except Exception as e:
        print(f"Feature extraction failed for {location}: {e}")
        return

    if df.empty:
        print(f"⚠ Empty DataFrame after processing {location}")
        return

    # Save ALL observations from this window
    save_stats(df)

    accel_count = df["accel"].notna().sum() if "accel" in df.columns else 0
    print(
        f"✅ Saved {location} | rows={len(df)} "
        f"| objects={df['object_id'].nunique()} "
        f"| accel_populated={accel_count}/{len(df)}"
    )
    
    # -------------------------
    # AUTO-RUN INCIDENT DETECTION
    # -------------------------
    try:
        print(f"Running incident detection for {location}...")
        
        # Load recent data (last 5 minutes should be enough)
        df_recent = load_all_combined_stats(limit=5000, location=location)
        
        if df_recent.empty:
            print(f"⚠ No data for incident detection")
            return
        
        # Scale and detect
        df_scaled, _ = scale_per_location(df_recent)
        models = load_latest_models()
        incidents = detect_incidents(df_scaled, models)
        
        if incidents:
            print(f"INCIDENTS DETECTED: {len(incidents)}")
            for incident in incidents:
                print(f"   - {incident}")
        else:
            print(f"No incidents detected")
            
    except Exception as e:
        print(f"Incident detection failed: {e}")


# =========================
# RAW INGESTION
# =========================

@app.post("/raw-vehicles")
def ingest_raw_vehicles(payload: List[Dict]):
    """
    TRUE FIXED WINDOW:
    - Accumulates data for exactly WINDOW_SECONDS
    - Processes entire buffer once window closes
    - Clears buffer and starts fresh
    """

    if not payload:
        return {"received": 0, "windows_closed": 0}

    now = datetime.utcnow()
    windows_closed = 0

    # -------------------------
    # Buffer incoming docs
    # -------------------------
    with BUFFER_LOCK:
        for doc in payload:
            location = doc.get("location", "unknown")

            # Initialize window if needed
            if location not in WINDOW_START:
                WINDOW_START[location] = now
                print(f"🆕 Started new window for {location}")

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
    # Check if windows should close
    # -------------------------
    locations_to_process = []
    
    with BUFFER_LOCK:
        for location in list(RAW_BUFFER.keys()):
            buf = RAW_BUFFER[location]
            if not buf:
                continue
            
            window_start = WINDOW_START.get(location)
            if window_start is None:
                continue
            
            # Calculate window duration
            elapsed = (now - window_start).total_seconds()
            
            # KEY: Only process if window is COMPLETE
            if elapsed >= WINDOW_SECONDS:
                print(f"⏰ Window closed for {location} after {elapsed:.1f}s | docs={len(buf)}")
                
                # Freeze buffer
                frozen_docs = list(buf)
                locations_to_process.append((location, frozen_docs))
                
                # Clear buffer and restart window
                RAW_BUFFER[location].clear()
                WINDOW_START[location] = now
                windows_closed += 1

    # -------------------------
    # Process closed windows asynchronously
    # -------------------------
    for location, docs in locations_to_process:
        Thread(
            target=process_window_async,
            args=(location, docs),
            daemon=True
        ).start()

    return {
        "received": len(payload),
        "windows_closed": windows_closed,
        "window_seconds": WINDOW_SECONDS,
    }

 
# =========================
# Incident detection
# =========================

@app.post("/incidents/run")
def run_incident_detection(location: str | None = None, limit: int = 10_000):
    """
    Runs incident detection on existing combined_stats.
    """

    df = load_all_combined_stats(limit=limit, location=location)

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