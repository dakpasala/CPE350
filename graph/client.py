#!/usr/bin/env python3
"""
client.py

Main launcher that starts BOTH:
1. WebSocket client (receives data from backend)
2. Map viewer (displays data on interactive map)

Usage: python3 client.py
"""

import asyncio
import threading
import time
import sys
from pathlib import Path

# Import the map viewer
import map_viewer

# Import WebSocket client logic
import websockets
import json
from datetime import datetime


# =========================
# Config
# =========================

BACKEND_WS_URL = "ws://127.0.0.1:8000/data/stream"
OUTPUT_FILE = Path(__file__).parent / "live_data.json"


# =========================
# WebSocket Client (runs in background thread)
# =========================

async def websocket_receiver():
    """
    Connect to backend WebSocket and save incoming data to JSON.
    Runs in background thread.
    """
    print(f"🔌 [WebSocket] Connecting to {BACKEND_WS_URL}...")
    
    while True:
        try:
            async with websockets.connect(BACKEND_WS_URL) as websocket:
                print(f"✅ [WebSocket] Connected to backend!")
                print(f"💾 [WebSocket] Saving data to: {OUTPUT_FILE}")
                print(f"🎯 [WebSocket] Waiting for 15-second windows...\n")
                
                # Send ping to keep connection alive
                await websocket.send("ping")
                
                while True:
                    # Receive message from backend
                    message = await websocket.recv()
                    
                    # Handle pong responses
                    if message == "pong":
                        continue
                    
                    # Parse JSON data
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError as e:
                        print(f"❌ [WebSocket] Failed to parse JSON: {e}")
                        continue
                    
                    # Check if it's window data
                    if data.get("type") != "window_complete":
                        print(f"⚠️  [WebSocket] Unknown message type: {data.get('type')}")
                        continue
                    
                    # Log received data
                    timestamp = data.get("timestamp", "unknown")
                    vehicle_count = data.get("vehicle_count", 0)
                    incident_count = data.get("incident_count", 0)
                    
                    print(f"📥 [WebSocket] Received window data:")
                    print(f"   Timestamp: {timestamp}")
                    print(f"   Vehicles: {vehicle_count}")
                    print(f"   Incidents: {incident_count}")
                    
                    # Save to JSON file
                    with open(OUTPUT_FILE, "w") as f:
                        json.dump(data, f, indent=2)
                    
                    print(f"   ✅ Saved to {OUTPUT_FILE}\n")
                    
                    # Send ping to keep alive
                    await websocket.send("ping")
        
        except websockets.exceptions.ConnectionClosed:
            print("❌ [WebSocket] Connection closed by server")
        except Exception as e:
            print(f"❌ [WebSocket] Error: {e}")
        
        print("🔄 [WebSocket] Reconnecting in 5 seconds...")
        await asyncio.sleep(5)


def run_websocket_client():
    """
    Run WebSocket client in asyncio event loop.
    This runs in a separate thread.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_receiver())


# =========================
# Main Launcher
# =========================

def main():
    print("=" * 70)
    print(" " * 15 + "🚗 LIVE TRAFFIC MONITORING CLIENT 🚗")
    print("=" * 70)
    print()
    print("Starting two components:")
    print("  1. WebSocket Client → Receives data from backend")
    print("  2. Map Viewer       → Displays data on interactive map")
    print()
    print("=" * 70)
    print()
    
    # Start WebSocket client in background thread
    print("🚀 [WebSocket] Starting background receiver...")
    ws_thread = threading.Thread(target=run_websocket_client, daemon=True)
    ws_thread.start()
    
    # Give WebSocket a moment to start
    time.sleep(2)
    
    # Start map viewer (this blocks - runs Dash server)
    print("🚀 [Map] Starting interactive map viewer...")
    print()
    
    try:
        map_viewer.main()
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down client...")
        sys.exit(0)


if __name__ == "__main__":
    main()