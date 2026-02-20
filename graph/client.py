#!/usr/bin/env python3
"""
client.py

Combined launcher with SHARED MEMORY (no JSON file).
WebSocket receives data -> stores in memory -> viewers read it.
Now includes THREE viewers:
1. Map viewer - animated vehicle tracking
2. Incident viewer - browse accident videos  
3. Heatmap viewer - traffic density visualization
"""

import asyncio
import threading
import time
import sys
import os
from pathlib import Path
from threading import Lock

# Add visualization directory to path for heatmap_viewer
sys.path.insert(0, str(Path(__file__).parent.parent / 'visualization'))

# Import viewers
import map_viewer
import incident_viewer
import heatmap_viewer

# WebSocket imports
import websockets
import json
from datetime import datetime


# =========================
# Config
# =========================

BACKEND_WS_URL = "ws://127.0.0.1:8000/data/stream"


# =========================
# SHARED MEMORY (instead of JSON file)
# =========================

SHARED_DATA = None
SHARED_DATA_LOCK = Lock()


def get_latest_data():
    """Get latest data from shared memory (called by all viewers)."""
    with SHARED_DATA_LOCK:
        return SHARED_DATA


def set_latest_data(data):
    """Set latest data in shared memory (called by WebSocket)."""
    global SHARED_DATA
    with SHARED_DATA_LOCK:
        SHARED_DATA = data


# =========================
# WebSocket Client
# =========================

async def websocket_receiver():
    """
    Connect to backend WebSocket and store incoming data in memory.
    """
    print(f"[WebSocket] Connecting to {BACKEND_WS_URL}...")
    
    while True:
        try:
            async with websockets.connect(BACKEND_WS_URL, ping_interval=20) as websocket:
                print(f"[WebSocket] Connected to backend!")
                print(f"[WebSocket] Waiting for 15-second windows...\n")
                
                while True:
                    # Receive message with timeout
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    except asyncio.TimeoutError:
                        # Keep alive
                        await websocket.send("ping")
                        continue
                    
                    # Handle pong
                    if message == "pong":
                        continue
                    
                    # Parse JSON data
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] [WebSocket] Failed to parse JSON: {e}")
                        continue
                    
                    # Check type
                    if data.get("type") != "window_complete":
                        print(f"[WARNING] [WebSocket] Unknown message type: {data.get('type')}")
                        continue
                    
                    # Log received data
                    timestamp = data.get("timestamp", "unknown")
                    vehicle_count = data.get("vehicle_count", 0)
                    incident_count = data.get("incident_count", 0)
                    
                    print(f"[RECEIVED] [WebSocket] Received window data:")
                    print(f"   Timestamp: {timestamp}")
                    print(f"   Vehicles: {vehicle_count}")
                    print(f"   Incidents: {incident_count}")
                    
                    # Store in SHARED MEMORY
                    set_latest_data(data)
                    
                    print(f"   [OK] Stored in memory\n")
        
        except websockets.exceptions.ConnectionClosed:
            print("[ERROR] [WebSocket] Connection closed by server")
        except Exception as e:
            print(f"[ERROR] [WebSocket] Error: {e}")
        
        print("[RETRY] [WebSocket] Reconnecting in 5 seconds...")
        await asyncio.sleep(5)


def run_websocket_client():
    """Run WebSocket client in asyncio event loop."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_receiver())


def run_incident_viewer():
    """Run incident viewer in background thread."""
    print("[START] [Incident Viewer] Starting on http://127.0.0.1:8051...")
    incident_viewer.main()


def run_heatmap_viewer():
    """Run heatmap viewer in background thread."""
    print("[START] [Heatmap Viewer] Starting on http://0.0.0.0:8052...")
    heatmap_viewer.main()


# =========================
# Main Launcher
# =========================

def main():
    print("=" * 70)
    print(" " * 15 + "LIVE TRAFFIC MONITORING CLIENT")
    print("=" * 70)
    print()
    print("Starting FOUR components:")
    print("  1. WebSocket Client    -> Receives data from backend")
    print("  2. Map Viewer          -> http://127.0.0.1:8050 ✅")
    print("  3. Incident Viewer     -> http://127.0.0.1:8051 ✅")
    print("  4. Heatmap Viewer      -> http://YOUR_IP:8052 🔥")
    print()
    print("Using SHARED MEMORY (no JSON file)")
    print("=" * 70)
    print()
    
    # Inject the get_latest_data function into all viewer modules
    map_viewer.get_latest_data = get_latest_data
    incident_viewer.get_latest_data = get_latest_data
    heatmap_viewer.get_latest_data = get_latest_data
    
    # Start WebSocket client in background thread
    print("[START] [WebSocket] Starting background receiver...")
    ws_thread = threading.Thread(target=run_websocket_client, daemon=True)
    ws_thread.start()
    
    # Give WebSocket a moment to start
    time.sleep(1)
    
    # Start incident viewer in background thread
    print("[START] [Incident Viewer] Starting background viewer...")
    incident_thread = threading.Thread(target=run_incident_viewer, daemon=True)
    incident_thread.start()
    
    # Give incident viewer a moment to start
    time.sleep(2)
    
    # Start heatmap viewer in background thread
    print("[START] [Heatmap Viewer] Starting background heatmap...")
    heatmap_thread = threading.Thread(target=run_heatmap_viewer, daemon=True)
    heatmap_thread.start()
    
    # Give heatmap viewer a moment to start
    time.sleep(2)
    
    # Display network information
    print("\n" + "=" * 70)
    print("🌐 ACCESS URLS:")
    print("=" * 70)
    try:
        import socket
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        print(f"  Map Viewer:      http://127.0.0.1:8050")
        print(f"  Incident Viewer: http://127.0.0.1:8051")
        print(f"  Heatmap Viewer:  http://{local_ip}:8052  <- ACCESS FROM ANY DEVICE")
        print(f"                   http://127.0.0.1:8052  <- LOCAL ACCESS")
    except:
        print(f"  Map Viewer:      http://127.0.0.1:8050")
        print(f"  Incident Viewer: http://127.0.0.1:8051")
        print(f"  Heatmap Viewer:  http://127.0.0.1:8052")
    print("=" * 70)
    print()
    
    # Start map viewer (this blocks - runs Dash server)
    print("[START] [Map Viewer] Starting interactive map viewer...")
    print()
    
    try:
        map_viewer.main()
    except KeyboardInterrupt:
        print("\n\n[EXIT] Shutting down client...")
        sys.exit(0)


if __name__ == "__main__":
    main()