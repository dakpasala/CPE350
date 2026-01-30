#!/usr/bin/env python3
"""
websocket_client.py

Connects to backend WebSocket, receives 15-second window data,
saves to JSON file for the map viewer.
"""

import asyncio
import websockets
import json
from datetime import datetime
from pathlib import Path


# =========================
# Config
# =========================

BACKEND_WS_URL = "ws://127.0.0.1:8000/data/stream"
OUTPUT_FILE = Path(__file__).parent / "live_data.json"


# =========================
# WebSocket Client
# =========================

async def connect_and_receive():
    """
    Connect to backend WebSocket and save incoming data to JSON.
    """
    print(f"🔌 Connecting to {BACKEND_WS_URL}...")
    
    try:
        async with websockets.connect(BACKEND_WS_URL) as websocket:
            print(f"✅ Connected to backend!")
            print(f"💾 Saving data to: {OUTPUT_FILE}")
            print(f"🎯 Waiting for 15-second windows...\n")
            
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
                    print(f"❌ Failed to parse JSON: {e}")
                    continue
                
                # Check if it's window data
                if data.get("type") != "window_complete":
                    print(f"⚠️  Received unknown message type: {data.get('type')}")
                    continue
                
                # Log received data
                timestamp = data.get("timestamp", "unknown")
                vehicle_count = data.get("vehicle_count", 0)
                incident_count = data.get("incident_count", 0)
                
                print(f"📥 Received window data:")
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
        print("❌ Connection closed by server")
    except Exception as e:
        print(f"❌ Error: {e}")


# =========================
# Main
# =========================

async def main():
    """
    Main loop - reconnect if connection drops.
    """
    while True:
        try:
            await connect_and_receive()
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            print("🔄 Reconnecting in 5 seconds...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    print("=" * 60)
    print("WebSocket Client - Traffic Data Receiver")
    print("=" * 60)
    print()
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")