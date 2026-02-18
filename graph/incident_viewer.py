#!/usr/bin/env python3
"""
incident_viewer.py

Incident video viewer - Browse and watch accident videos from MongoDB.
Fetches incidents with associated videos and displays them.
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

import dash
from dash import Dash, dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

load_dotenv()

# =========================
# Configuration
# =========================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
HOST = os.getenv("VIEWER_HOST", "127.0.0.1")
PORT = int(os.getenv("VIEWER_PORT", "8051"))

TIME_RANGES = {
    "Last Hour": 60,
    "Last 6 Hours": 360,
    "Last 12 Hours": 720,
    "Last Day": 1440,
    "Last Week": 10080,
    "Last Month": 43200,
}


# =========================
# API Calls
# =========================

def get_recent_videos(limit=10, location=None):
    """Fetch recent videos from API."""
    try:
        params = {"limit": limit}
        if location:
            params["location"] = location
        
        response = requests.get(f"{API_BASE_URL}/videos", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("videos", [])
        else:
            print(f"⚠️ Failed to fetch videos: {response.status_code}")
            return []
    except Exception as e:
        print(f"⚠️ Error fetching videos: {e}")
        return []


def get_videos_by_timerange(minutes, location=None):
    """Fetch videos from last N minutes."""
    try:
        params = {"minutes": minutes}
        if location:
            params["location"] = location
        
        response = requests.get(f"{API_BASE_URL}/videos/timerange", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("videos", [])
        else:
            print(f"⚠️ Failed to fetch videos: {response.status_code}")
            return []
    except Exception as e:
        print(f"⚠️ Error fetching videos: {e}")
        return []


def get_recent_incidents(limit=50, location=None):
    """Fetch recent incidents from API."""
    try:
        params = {"limit": limit}
        if location:
            params["location"] = location
        
        response = requests.get(f"{API_BASE_URL}/incidents/recent", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("incidents", [])
        else:
            print(f"⚠️ Failed to fetch incidents: {response.status_code}")
            return []
    except Exception as e:
        print(f"⚠️ Error fetching incidents: {e}")
        return []


def get_incidents_by_timerange(minutes, location=None):
    """Fetch incidents from last N minutes."""
    try:
        params = {"minutes": minutes}
        if location:
            params["location"] = location
        
        response = requests.get(f"{API_BASE_URL}/incidents/timerange", params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("incidents", [])
        else:
            print(f"⚠️ Failed to fetch incidents: {response.status_code}")
            return []
    except Exception as e:
        print(f"⚠️ Error fetching incidents: {e}")
        return []


def get_video_for_incident(incident_id):
    """Fetch video associated with an incident."""
    try:
        response = requests.get(f"{API_BASE_URL}/videos/incident/{incident_id}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("status") == "success":
                return data.get("video")
        return None
    except Exception as e:
        print(f"⚠️ Error fetching video for incident: {e}")
        return None


# =========================
# Helper Functions
# =========================

def format_timestamp(ts_str):
    """Format timestamp for display."""
    try:
        if isinstance(ts_str, str):
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        else:
            ts = ts_str
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return str(ts_str)


def build_video_card(video):
    """Build a card for a single video."""
    video_id = video.get("_id")
    filename = video.get("filename", "Unknown")
    camera = video.get("camera", "Unknown")
    timestamp = format_timestamp(video.get("timestamp"))
    size_mb = video.get("size_bytes", 0) / 1024 / 1024
    incident_count = len(video.get("incident_ids", []))
    
    # Determine card color based on incidents
    card_color = "danger" if incident_count > 0 else "secondary"
    
    return dbc.Card([
        dbc.CardBody([
            html.H5(f"📹 {filename}", className="card-title"),
            html.P([
                html.Strong("Camera: "), camera, html.Br(),
                html.Strong("Time: "), timestamp, html.Br(),
                html.Strong("Size: "), f"{size_mb:.2f} MB", html.Br(),
                html.Strong("Incidents: "), 
                html.Span(
                    f"🚨 {incident_count}" if incident_count > 0 else "None",
                    style={"color": "red" if incident_count > 0 else "green"}
                ),
            ]),
            dbc.Button(
                "▶️ Watch Video",
                id={"type": "watch-btn", "index": video_id},
                color="primary",
                className="mt-2"
            ),
        ])
    ], color=card_color, outline=True, className="mb-3")


def build_incident_card(incident):
    """Build a card for a single incident."""
    incident_id = str(incident.get("_id"))
    incident_type = incident.get("incident_type", "Unknown")
    severity = incident.get("severity", 0)
    timestamp = format_timestamp(incident.get("timestamp"))
    vehicles = incident.get("vehicles", [])
    location = incident.get("location", "Unknown")
    
    return dbc.Card([
        dbc.CardBody([
            html.H5(f"🚨 {incident_type.upper()}", className="card-title"),
            html.P([
                html.Strong("Severity: "), f"{severity:.2f}", html.Br(),
                html.Strong("Time: "), timestamp, html.Br(),
                html.Strong("Location: "), location, html.Br(),
                html.Strong("Vehicles: "), f"{len(vehicles)}", html.Br(),
            ]),
            dbc.Button(
                "🎥 Watch Video",
                id={"type": "incident-video-btn", "index": incident_id},
                color="danger",
                className="mt-2"
            ),
        ])
    ], color="danger", outline=True, className="mb-3")


# =========================
# Dash App
# =========================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Incident Video Viewer"

app.layout = dbc.Container([
    html.H1("🚨 Traffic Incident Video Viewer", className="mt-4 mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.A(
                dbc.Button("← Back to Live Feed", color="secondary", className="mb-3"),
                href="http://127.0.0.1:8050"
            ),
        ]),
    ]),
    
    # Tabs for different views
    dbc.Tabs([
        # Tab 1: Incidents by Time Range with Videos
        dbc.Tab([
            html.Div([
                html.H3("Incidents by Time Range", className="mt-4"),
                html.P("Browse incidents from a specific time period and watch videos"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Time Range:"),
                        dbc.Select(
                            id="incident-timerange-select",
                            options=[{"label": k, "value": v} for k, v in TIME_RANGES.items()],
                            value=1440,  # Default: Last Day
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Location:"),
                        dbc.Input(
                            id="incident-location-input",
                            placeholder="e.g., patterson (optional)",
                            type="text"
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Br(),
                        dbc.Button("🔍 Search Incidents", id="search-incidents-btn", color="danger"),
                    ], width=4),
                ], className="mb-4"),
                
                html.Div(id="incidents-container"),
            ])
        ], label="🚨 Incidents by Time"),
        
        # Tab 2: Browse All Videos
        dbc.Tab([
            html.Div([
                html.H3("Browse All Videos", className="mt-4"),
                
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Time Range:"),
                        dbc.Select(
                            id="timerange-select",
                            options=[{"label": k, "value": v} for k, v in TIME_RANGES.items()],
                            value=1440,  # Default: Last Day
                        ),
                    ], width=4),
                    dbc.Col([
                        dbc.Label("Location:"),
                        dbc.Input(
                            id="location-input",
                            placeholder="e.g., patterson (optional)",
                            type="text"
                        ),
                    ], width=4),
                    dbc.Col([
                        html.Br(),
                        dbc.Button("🔍 Search Videos", id="search-videos-btn", color="primary"),
                    ], width=4),
                ], className="mb-4"),
                
                html.Div(id="videos-container"),
            ])
        ], label="📹 All Videos"),
        
        # Tab 3: Latest Video (for quick access)
        dbc.Tab([
            html.Div([
                html.H3("Latest Captured Video", className="mt-4"),
                html.P("Most recent video from the camera"),
                
                dbc.Button("🔄 Refresh", id="refresh-latest-btn", color="primary", className="mb-3"),
                
                html.Div(id="latest-video-container"),
            ])
        ], label="🎬 Latest Video"),
    ]),
    
    # Video Player Modal
    dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Video Player")),
        dbc.ModalBody([
            html.Div(id="video-player-content"),
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-video-modal", className="ml-auto")
        ),
    ], id="video-modal", size="xl", is_open=False),
    
], fluid=True)


# =========================
# Callbacks
# =========================

@app.callback(
    Output("incidents-container", "children"),
    Input("search-incidents-btn", "n_clicks"),
    State("incident-timerange-select", "value"),
    State("incident-location-input", "value"),
    prevent_initial_call=False,
)
def search_incidents(n_clicks, minutes, location):
    """Search and display incidents by time range."""
    location = location.strip() if location else None
    incidents = get_incidents_by_timerange(minutes, location)
    
    if not incidents:
        return dbc.Alert("No incidents found for this time range", color="info")
    
    cards = [build_incident_card(inc) for inc in incidents]
    
    return html.Div([
        html.P(f"Found {len(incidents)} incidents in the last {minutes} minutes", style={"fontSize": "18px", "fontWeight": "bold"}),
        dbc.Row([dbc.Col(card, width=4) for card in cards]),
    ])


@app.callback(
    Output("videos-container", "children"),
    Input("search-videos-btn", "n_clicks"),
    State("timerange-select", "value"),
    State("location-input", "value"),
    prevent_initial_call=False,
)
def search_videos(n_clicks, minutes, location):
    """Search and display videos by time range."""
    location = location.strip() if location else None
    videos = get_videos_by_timerange(minutes, location)
    
    if not videos:
        return dbc.Alert("No videos found for this time range", color="info")
    
    cards = [build_video_card(video) for video in videos]
    
    return html.Div([
        html.P(f"Found {len(videos)} videos"),
        dbc.Row([dbc.Col(card, width=4) for card in cards]),
    ])


@app.callback(
    Output("latest-video-container", "children"),
    Input("refresh-latest-btn", "n_clicks"),
    prevent_initial_call=False,
)
def load_latest_video(n_clicks):
    """Load the most recent video."""
    videos = get_recent_videos(limit=1)
    
    if not videos:
        return dbc.Alert("No videos found", color="warning")
    
    video = videos[0]
    video_id = video.get("_id")
    
    return html.Div([
        build_video_card(video),
        html.Hr(),
        html.H5("Preview:"),
        html.Video(
            src=f"{API_BASE_URL}/videos/{video_id}",
            controls=True,
            style={"width": "100%", "maxWidth": "800px"}
        ),
    ])


@app.callback(
    Output("video-modal", "is_open"),
    Output("video-player-content", "children"),
    Input({"type": "watch-btn", "index": dash.ALL}, "n_clicks"),
    Input({"type": "incident-video-btn", "index": dash.ALL}, "n_clicks"),
    Input("close-video-modal", "n_clicks"),
    State("video-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_video_modal(watch_clicks, incident_clicks, close_click, is_open):
    """Open/close video player modal."""
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return False, ""
    
    trigger = ctx.triggered[0]
    prop_id = trigger["prop_id"]
    
    # Close button
    if "close-video-modal" in prop_id:
        return False, ""
    
    # Watch button (direct video)
    if "watch-btn" in prop_id:
        import json
        button_id = json.loads(prop_id.split(".")[0])
        video_id = button_id["index"]
        
        video_player = html.Video(
            src=f"{API_BASE_URL}/videos/{video_id}",
            controls=True,
            autoPlay=True,
            style={"width": "100%"}
        )
        
        return True, video_player
    
    # Incident video button (fetch video for incident)
    if "incident-video-btn" in prop_id:
        import json
        button_id = json.loads(prop_id.split(".")[0])
        incident_id = button_id["index"]
        
        video_info = get_video_for_incident(incident_id)
        
        if video_info:
            video_id = video_info.get("_id")
            
            video_player = html.Div([
                html.H5(f"Video: {video_info.get('filename')}"),
                html.P(f"Camera: {video_info.get('camera')} | Time: {format_timestamp(video_info.get('timestamp'))}"),
                html.Video(
                    src=f"{API_BASE_URL}/videos/{video_id}",
                    controls=True,
                    autoPlay=True,
                    style={"width": "100%"}
                ),
            ])
            
            return True, video_player
        else:
            return True, dbc.Alert("No video found for this incident", color="warning")
    
    return False, ""


# =========================
# Main
# =========================

def main():
    print("\n" + "=" * 60)
    print("Incident Video Viewer")
    print("=" * 60)
    print(f"📍 Open: http://{HOST}:{PORT}")
    print(f"🔗 API: {API_BASE_URL}")
    print("=" * 60 + "\n")
    
    # Disable debug mode when running in thread (signals don't work in threads)
    app.run(debug=False, host=HOST, port=PORT)


if __name__ == "__main__":
    main()