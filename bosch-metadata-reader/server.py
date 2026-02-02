from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from datetime import datetime, timedelta
from typing import List, Dict

from collections import defaultdict, deque
import pandas as pd
from threading import Thread, Lock
import json
import asyncio


# =========================
# Local imports
# =========================

from combined_statistics import ( process_raw_docs, save_stats )

from incident_detection.engine import detect_incidents
from incident_detection.models import load_latest_models
from incident_detection.data import scale_per_location
from email_alert import send_incident_email

from data import (
    save_incidents, 
    get_recent_incidents, 
    get_incident_statistics,
    get_incidents_by_timerange,
    load_all_combined_stats,
    delete_combined_stats,
    delete_old_incidents
)


# =========================
# App init
# =========================

app = FastAPI(
    title="Traffic Incident Detection API",
    version="1.0"
)


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
# WebSocket config
# =========================

# WebSocket connections (for real-time incident push)
ACTIVE_WEBSOCKETS: List[WebSocket] = []
WEBSOCKET_LOCK = Lock()

# WebSocket connections for vehicle position streaming
VEHICLE_STREAM_CLIENTS: List[WebSocket] = []
VEHICLE_STREAM_LOCK = Lock()

# Message queue for sending data to WebSocket clients
MESSAGE_QUEUE = []
MESSAGE_QUEUE_LOCK = Lock()


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
# Raw ingestion
# =========================

@app.post("/raw-vehicles")
def ingest_raw_vehicles(payload: List[Dict]):
    """
    TRUE FIXED WINDOW:
    - Accumulates data for exactly WINDOW_SECONDS
    - Processes entire buffer once window closes
    - Sends enriched data + incidents to frontend together
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
                print(f"[NEW] Started new window for {location}")

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
                print(f"[WINDOW] Window closed for {location} after {elapsed:.1f}s | docs={len(buf)}")
                
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
# Window processing
# =========================

def process_window_async(location: str, docs: list[Dict]):
    """
    Processes one complete 15-second window.
    Then sends BOTH vehicle data + incidents to frontend together.
    """
    
    if not docs:
        return
    
    print(f"[Processing] {location} window | docs={len(docs)}")
    
    try:
        # Feature extraction on full window
        df = process_raw_docs(docs)
    except Exception as e:
        print(f"[ERROR] Feature extraction failed for {location}: {e}")
        return

    if df.empty:
        print(f"[WARNING] Empty DataFrame after processing {location}")
        return

    # Save to DB for historical analysis
    save_stats(df)

    accel_count = df["accel"].notna().sum() if "accel" in df.columns else 0
    print(
        f"[SAVED] {location} | rows={len(df)} "
        f"| objects={df['object_id'].nunique()} "
        f"| accel_populated={accel_count}/{len(df)}"
    )
    
    # -------------------------
    # RUN INCIDENT DETECTION
    # -------------------------
    incidents = []
    try:
        print(f"[DETECTION] Running incident detection for {location}...")
        
        # Use the SAME dataframe we just processed (not from DB)
        df_scaled, _ = scale_per_location(df)
        models = load_latest_models()
        incidents = detect_incidents(df_scaled, models)
        
        if incidents:
            print(f"[INCIDENTS] DETECTED: {len(incidents)}")
            
            # Save incidents to MongoDB
            saved_count = save_incidents(incidents)
            print(f"[DB] Saved {saved_count} incidents to DB")
            
            # Log details
            for incident in incidents:
                print(f"   - {incident['incident_type']} | severity={incident['severity']:.2f} | vehicles={incident['vehicles']}")
        else:
            print(f"[OK] No incidents detected")
            
    except Exception as e:
        print(f"[ERROR] Incident detection failed: {e}")
    
    # -------------------------
    # SEND ENRICHED DATA TO FRONTEND
    # -------------------------
    # Create a mapping of which vehicles are in incidents
    incident_vehicle_map = {}
    for incident in incidents:
        for vehicle_id in incident.get("vehicles", []):
            if vehicle_id not in incident_vehicle_map:
                incident_vehicle_map[vehicle_id] = []
            incident_vehicle_map[vehicle_id].append({
                "type": incident["incident_type"],
                "severity": incident["severity"],
                "timestamp": incident["timestamp"]
            })
    
    # Convert DataFrame to frontend-ready format
    vehicles = []
    for _, row in df.iterrows():
        vehicle_id = str(row["object_id"])
        
        vehicle_data = {
            "id": vehicle_id,
            "lat": float(row["lat"]) if pd.notna(row["lat"]) else None,
            "lon": float(row["lon"]) if pd.notna(row["lon"]) else None,
            "speed_mps": float(row["speed_mps"]) if pd.notna(row["speed_mps"]) else 0,
            "heading_deg": float(row["heading_deg"]) if pd.notna(row["heading_deg"]) else 0,
            "accel": float(row["accel"]) if pd.notna(row["accel"]) else None,
            "timestamp": row["timestamp"].isoformat() if pd.notna(row["timestamp"]) else None,
            "detected_type": row["detected_type"],
            "location": row["location"],
            "certainty": float(row["certainty"]) if pd.notna(row["certainty"]) else 0,
            # INCLUDE INCIDENT DATA IF THIS VEHICLE IS INVOLVED
            "incidents": incident_vehicle_map.get(int(vehicle_id), [])
        }
        
        vehicles.append(vehicle_data)
    
    # Send combined data to frontend
    broadcast_window_data(location, vehicles, incidents)


# =========================
# Broadcasting
# =========================

def broadcast_window_data(location: str, vehicles: List[Dict], incidents: List[Dict]):
    """
    Send complete 15-second window data to frontend:
    - All vehicle observations (with features)
    - All detected incidents
    - Mapping of which vehicles are in incidents
    """
    
    message = {
        "type": "window_complete",
        "location": location,
        "timestamp": datetime.utcnow().isoformat(),
        "vehicle_count": len(vehicles),
        "incident_count": len(incidents),
        "vehicles": vehicles,
        "incidents": incidents
    }
    
    # PRINT DATA BEING SENT
    print(f"\n[FRONTEND] DATA PREPARED FOR FRONTEND:")
    print(f"   Location: {location}")
    print(f"   Timestamp: {message['timestamp']}")
    print(f"   Vehicle count: {len(vehicles)}")
    print(f"   Incident count: {len(incidents)}")
    
    if vehicles:
        print(f"\n   [SAMPLE] Sample vehicles (showing first 3):")
        for v in vehicles[:3]:
            print(f"      ID={v['id']} | lat={v['lat']:.6f}, lon={v['lon']:.6f}")
            print(f"        speed={v['speed_mps']:.2f} m/s, accel={v['accel']}, heading={v['heading_deg']:.1f} deg")
            print(f"        type={v['detected_type']}, certainty={v['certainty']:.2f}")
            if v['incidents']:
                print(f"        [INCIDENT] INVOLVED IN INCIDENTS: {v['incidents']}")
    
    if incidents:
        print(f"\n   [INCIDENTS] Incidents detected:")
        for inc in incidents:
            print(f"      {inc['incident_type']} | severity={inc['severity']:.2f}")
            print(f"        vehicles={inc['vehicles']} | time={inc['timestamp']}")
    
    # Check if any clients connected
    if not VEHICLE_STREAM_CLIENTS:
        print(f"\n   [WARNING] No clients connected - data NOT sent\n")
        return
    
    # Send to all connected clients
    print(f"\n   [OK] Sending to {len(VEHICLE_STREAM_CLIENTS)} client(s)")
    
    # Add message to queue - WebSocket will pick it up
    with MESSAGE_QUEUE_LOCK:
        MESSAGE_QUEUE.append(message)
        print(f"   [QUEUE] Added to message queue (queue size: {len(MESSAGE_QUEUE)})\n")


def broadcast_vehicle_positions(payload: List[Dict]):
    """
    Stream raw vehicle positions to frontend WebSocket clients in real-time.
    This sends data IMMEDIATELY, not waiting for 15-second window.
    """
    if not VEHICLE_STREAM_CLIENTS:
        return
    
    # Extract only what frontend needs for map rendering
    vehicles = []
    for doc in payload:
        map_path = doc.get("mapPath", [])
        if not map_path or not isinstance(map_path, list):
            continue
        
        # Get first position from mapPath
        if len(map_path) > 0 and isinstance(map_path[0], (list, tuple)) and len(map_path[0]) == 2:
            lat, lon = map_path[0]
            vehicles.append({
                "id": doc.get("id"),
                "lat": lat,
                "lon": lon,
                "speed": doc.get("speed", 0),
                "type": doc.get("detected_type", "unknown"),
                "timestamp": doc.get("timestamp"),
                "location": doc.get("location"),
                "certainty": doc.get("detection_certainty", 0)
            })
    
    if not vehicles:
        return
    
    with VEHICLE_STREAM_LOCK:
        message = {
            "type": "vehicles",
            "count": len(vehicles),
            "data": vehicles
        }
        
        disconnected = []
        for ws in VEHICLE_STREAM_CLIENTS:
            try:
                asyncio.create_task(ws.send_json(message))
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            VEHICLE_STREAM_CLIENTS.remove(ws)


def broadcast_incidents(incidents: List[Dict]):
    """
    Push incidents to all connected WebSocket clients in real-time.
    """
    if not ACTIVE_WEBSOCKETS:
        return
    
    with WEBSOCKET_LOCK:
        message = {
            "type": "incidents",
            "count": len(incidents),
            "data": incidents
        }
        
        disconnected = []
        for ws in ACTIVE_WEBSOCKETS:
            try:
                # Use asyncio to send (must be called from async context)
                asyncio.create_task(ws.send_json(message))
            except Exception:
                disconnected.append(ws)
        
        # Remove disconnected clients
        for ws in disconnected:
            ACTIVE_WEBSOCKETS.remove(ws)


# =========================
# Incident endpoints
# =========================

@app.post("/incidents/run")
def run_incident_detection(location: str | None = None, limit: int = 10_000):
    """
    Manually runs incident detection on existing combined_stats.
    (Note: This also runs automatically after each window)
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
    
    # Save to DB
    saved_count = save_incidents(incidents) if incidents else 0

    return {
        "rows_analyzed": len(df),
        "count": len(incidents),
        "saved": saved_count,
        "incidents": incidents
    }


@app.get("/incidents/recent")
def get_recent_incidents_endpoint(
    limit: int = 100,
    location: str | None = None,
    incident_type: str | None = None,
    min_severity: float | None = None
):
    """
    Retrieves recent incidents from the database.
    
    Query params:
    - limit: Max number of incidents (default 100)
    - location: Filter by camera location
    - incident_type: Filter by type ("collision" or "near_miss")
    - min_severity: Minimum severity threshold (0.0-1.0)
    """
    incidents = get_recent_incidents(
        limit=limit,
        location=location,
        incident_type=incident_type,
        min_severity=min_severity
    )
    
    return {
        "count": len(incidents),
        "incidents": incidents
    }


@app.get("/incidents/stats")
def get_incident_stats_endpoint(location: str | None = None):
    """
    Get summary statistics about detected incidents.
    """
    stats = get_incident_statistics(location=location)
    
    return stats


@app.get("/incidents/timerange")
def get_incidents_by_time(
    minutes: int = 15,
    location: str | None = None
):
    """
    Get incidents from the last N minutes.
    
    Query params:
    - minutes: How many minutes back to query (default 15)
    - location: Filter by camera location
    
    Example: GET /incidents/timerange?minutes=30&location=patterson
    """
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=minutes)
    
    incidents = get_incidents_by_timerange(start_time, end_time, location)
    
    return {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "minutes": minutes,
        "count": len(incidents),
        "incidents": incidents
    }


@app.delete("/incidents")
def delete_incidents_endpoint(days_old: int = 30):
    """
    Deletes incidents older than N days.

    Query params:
    - days_old: Delete incidents older than this many days (default 30)
    """
    deleted = delete_old_incidents(days_old=days_old)

    return {
        "deleted": deleted,
        "days_old": days_old
    }


# =========================
# Combined stats endpoints
# =========================

@app.get("/stats/combined")
def get_combined_stats(
    time_range: str = "day",
    limit: int = 10_000,
    location: str | None = None
):
    """
    Retrieve combined stats within a time range.

    Query params:
    - time_range: "hour", "6hours", "12hours", "day", "week", "month"
    - limit: Max rows to return (default 10,000)
    - location: Filter by location (e.g. "patterson")
    """
    df = load_all_combined_stats(time_range=time_range, limit=limit, location=location)

    if df.empty:
        return {"count": 0, "time_range": time_range, "data": []}

    # Convert to JSON-safe format (Timestamps -> ISO strings)
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return {
        "count": len(df),
        "time_range": time_range,
        "location": location,
        "data": df.to_dict(orient="records")
    }


@app.delete("/stats/combined")
def delete_combined_stats_endpoint(time_range: str = "day"):
    """
    Deletes combined_stats data within a time range.

    Query params:
    - time_range: "hour", "6hours", "12hours", "day", "week", "month"
    """
    deleted = delete_combined_stats(time_range=time_range)

    return {
        "deleted": deleted,
        "time_range": time_range
    }


# =========================
# WebSocket streaming
# =========================

@app.websocket("/data/stream")
async def data_stream(websocket: WebSocket):
    """
    WebSocket endpoint for complete 15-second window data.
    """
    await websocket.accept()
    
    with VEHICLE_STREAM_LOCK:
        VEHICLE_STREAM_CLIENTS.append(websocket)
    
    print(f"[WS] Data stream connected | total clients: {len(VEHICLE_STREAM_CLIENTS)}")
    print(f"[WS] Starting message loop...")
    
    loop_count = 0
    try:
        while True:
            loop_count += 1
            
            # Heartbeat every 50 loops (~2.5 seconds)
            if loop_count % 50 == 0:
                print(f"   [HEARTBEAT] WebSocket loop alive (iteration {loop_count})")
            
            # Check message queue for new data to send
            message_to_send = None
            with MESSAGE_QUEUE_LOCK:
                if MESSAGE_QUEUE:
                    message_to_send = MESSAGE_QUEUE.pop(0)
                    print(f"   [QUEUE] Found message in queue (remaining: {len(MESSAGE_QUEUE)})")
            
            if message_to_send:
                try:
                    await websocket.send_json(message_to_send)
                    print(f"   [OK] Sent message to client!")
                except Exception as e:
                    print(f"   [ERROR] Failed to send: {e}")
                    break
            
            # Handle ping/pong with timeout
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=0.05)
                if data == "ping":
                    await websocket.send_text("pong")
                    print(f"   [PONG] Pong sent")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"   [WARNING] Receive error: {e}")
                break
            
            await asyncio.sleep(0.05)
    
    except WebSocketDisconnect:
        print(f"[WS] Data stream disconnected")
    except Exception as e:
        print(f"[WS] Data stream error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        with VEHICLE_STREAM_LOCK:
            if websocket in VEHICLE_STREAM_CLIENTS:
                VEHICLE_STREAM_CLIENTS.remove(websocket)
        print(f"[WS] Client removed | remaining: {len(VEHICLE_STREAM_CLIENTS)}")