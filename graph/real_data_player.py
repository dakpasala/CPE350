#!/usr/bin/env python3
"""
real_data_replayer.py

Reads real vehicle data from data.txt and replays it to client.py via WebSocket.
Acts as a fake server that formats real camera data for the client.

Usage:
    1. Put your real data in data.txt (one JSON object per line)
    2. Stop your real server.py
    3. Run: python3 real_data_replayer.py
    4. Run: python3 client.py (it will connect to this replayer)

Data format expected in data.txt (one JSON per line):
{"location":"patterson1","data":{"timestamp":"...","objects":[...]}}
"""

import asyncio
import websockets
import json
import time
from datetime import datetime, timedelta
from collections import defaultdict

# Server settings (same as real server.py)
HOST = "127.0.0.1"
PORT = 8000
WS_PATH = "/data/stream"

# Store connected clients
CONNECTED_CLIENTS = set()

# Data file
DATA_FILE = "data.txt"


def parse_real_data_line(line):
    """
    Parse a line from data.txt and convert to our format.
    
    Input format:
    {"location":"patterson1","data":{"timestamp":"...","objects":[{"id":"...","type":"Car",...}]}}
    
    Output format (what client expects):
    {
        "type": "window_complete",
        "location": "patterson1",
        "timestamp": "...",
        "vehicles": [...],
        "incidents": []
    }
    """
    try:
        raw = json.loads(line.strip())
        location = raw.get("location", "unknown")
        data = raw.get("data", {})
        timestamp = data.get("timestamp")
        objects = data.get("objects", [])
        
        # Convert objects to our vehicle format
        vehicles = []
        for obj in objects:
            # Extract lat/lon from location array [lat, lon]
            loc = obj.get("location", [0, 0])
            lat = loc[0] if len(loc) > 0 else 0
            lon = loc[1] if len(loc) > 1 else 0
            
            vehicle = {
                "id": obj.get("id"),
                "location": location,
                "lat": lat,
                "lon": lon,
                "detected_type": obj.get("type", "Car").lower(),
                "certainty": obj.get("certainty", 0.5),
                "timestamp": timestamp,
                "speed_mps": obj.get("speed", 0),
                "accel": 0,  # Not in raw data, set to 0
                "incidents": [],
                "zones": [obj.get("zone", "unknown")] if obj.get("zone") else [],
            }
            vehicles.append(vehicle)
        
        return {
            "location": location,
            "timestamp": timestamp,
            "vehicles": vehicles
        }
    
    except Exception as e:
        print(f"[ERROR] Failed to parse line: {e}")
        return None


def load_real_data(filename):
    """
    Load all data from data.txt and group by location and time window.
    
    Returns: dict of {location: [windows]}
    where each window is a 15-second batch of vehicles
    """
    print(f"[LOADER] Reading data from {filename}...")
    
    # Group by location first
    location_data = defaultdict(list)
    
    try:
        with open(filename, 'r') as f:
            for line_num, line in enumerate(f, 1):
                if not line.strip():
                    continue
                
                parsed = parse_real_data_line(line)
                if parsed:
                    location = parsed["location"]
                    location_data[location].append(parsed)
                
                if line_num % 100 == 0:
                    print(f"[LOADER] Processed {line_num} lines...")
    
    except FileNotFoundError:
        print(f"[ERROR] File not found: {filename}")
        print(f"[ERROR] Please create {filename} with your real data (one JSON per line)")
        return {}
    
    print(f"[LOADER] Loaded data for {len(location_data)} locations")
    for loc, data in location_data.items():
        print(f"  - {loc}: {len(data)} data points")
    
    # Group into 15-second windows
    windows_by_location = {}
    
    for location, data_points in location_data.items():
        # Sort by timestamp
        data_points.sort(key=lambda x: x["timestamp"])
        
        # Group into windows (collect ~15 seconds worth)
        windows = []
        current_window = []
        window_start = None
        
        for point in data_points:
            if not window_start:
                window_start = point["timestamp"]
                current_window = [point]
            else:
                current_window.append(point)
                
                # After collecting some data, create a window
                if len(current_window) >= 15:  # ~15 data points = ~15 seconds
                    windows.append(current_window)
                    current_window = []
                    window_start = None
        
        # Add remaining data as final window
        if current_window:
            windows.append(current_window)
        
        windows_by_location[location] = windows
        print(f"  - {location}: Split into {len(windows)} windows")
    
    return windows_by_location


def create_window_message(location, window_data):
    """
    Create a window_complete message from a batch of data points.
    """
    # Combine all vehicles from the window
    all_vehicles = []
    for point in window_data:
        all_vehicles.extend(point["vehicles"])
    
    # Use first timestamp as window timestamp
    timestamp = window_data[0]["timestamp"] if window_data else datetime.now().isoformat()
    
    return {
        "type": "window_complete",
        "location": location,
        "timestamp": timestamp,
        "vehicle_count": len(all_vehicles),
        "incident_count": 0,  # No incidents in raw data
        "vehicles": all_vehicles,
        "incidents": [],
    }


async def broadcast_to_clients(message):
    """Send message to all connected clients."""
    if CONNECTED_CLIENTS:
        tasks = [client.send(message) for client in CONNECTED_CLIENTS]
        await asyncio.gather(*tasks, return_exceptions=True)


async def replay_data(windows_by_location):
    """
    Replay real data to connected clients.
    Sends one window per location every 15 seconds (looping).
    """
    if not windows_by_location:
        print("[REPLAYER] No data to replay!")
        return
    
    print("[REPLAYER] Starting data replay...")
    
    # Track which window we're on for each location
    window_indices = {loc: 0 for loc in windows_by_location.keys()}
    
    # Stagger locations by a few seconds
    location_offsets = {}
    for i, loc in enumerate(sorted(windows_by_location.keys())):
        location_offsets[loc] = i * 3  # 3 second stagger
    
    while True:
        current_time = time.time()
        
        for location, windows in windows_by_location.items():
            offset = location_offsets[location]
            
            # Check if it's time to send data for this location
            if (current_time - offset) % 15 < 1:  # Every 15 seconds
                window_idx = window_indices[location]
                window_data = windows[window_idx]
                
                # Create message
                message = create_window_message(location, window_data)
                
                print(f"[{location.upper()}] Window #{window_idx + 1}/{len(windows)}: {message['vehicle_count']} vehicles")
                
                # Broadcast to clients
                await broadcast_to_clients(json.dumps(message))
                
                # Move to next window (loop back to start if at end)
                window_indices[location] = (window_idx + 1) % len(windows)
        
        await asyncio.sleep(1)


async def handle_client(websocket):
    """Handle a client connection."""
    print(f"[CLIENT] New connection from {websocket.remote_address}")
    
    # Add to connected clients
    CONNECTED_CLIENTS.add(websocket)
    
    try:
        # Keep connection alive
        async for message in websocket:
            # Handle ping/pong
            if message == "ping":
                await websocket.send("pong")
    
    except websockets.exceptions.ConnectionClosed:
        print(f"[CLIENT] Connection closed")
    
    finally:
        # Remove from connected clients
        CONNECTED_CLIENTS.discard(websocket)


async def main():
    print("=" * 70)
    print(" " * 15 + "REAL DATA REPLAYER SERVER")
    print("=" * 70)
    print()
    
    # Load real data from file
    windows_by_location = load_real_data(DATA_FILE)
    
    if not windows_by_location:
        print("\n[ERROR] No data loaded. Exiting.")
        return
    
    print()
    print(f"WebSocket Server: ws://{HOST}:{PORT}{WS_PATH}")
    print()
    print("Waiting for client.py to connect...")
    print("Press Ctrl+C to stop")
    print("=" * 70)
    print()
    
    # Start WebSocket server
    server = await websockets.serve(handle_client, HOST, PORT)
    
    # Start data replay in background
    replay_task = asyncio.create_task(replay_data(windows_by_location))
    
    # Keep running
    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[EXIT] Stopping replayer...")