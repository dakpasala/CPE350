#!/usr/bin/env python3
"""
map_viewer.py

Continuous live feed with 15-second animated playback.
Starts immediately with empty map, populates when data arrives.
Enhanced UI with video playback on incident detection.
"""

import os
import math
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import time

import pandas as pd
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


# =========================
# Data source (injected by client.py)
# =========================

# This function will be replaced by client.py with actual shared memory accessor
def get_latest_data():
    """Placeholder - replaced by client.py at runtime."""
    return None

load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_ACCESS_TOKEN is missing in .env")

ANCHOR = {"lat": 34.441560, "lon": -119.808362}
API_BASE_URL = "http://localhost:8000"

MAP_STYLE = "mapbox://styles/mapbox/satellite-streets-v12"
DEFAULT_ZOOM = 18

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8050"))

# Play through 15 seconds in ~15 seconds (1x speed)
PLAYBACK_FPS = 3  # Match actual data rate
FRAME_INTERVAL_MS = int(1000 / 3)  # ~333ms per frame

COLOR_MAP = {
    "car": "rgb(255,0,0)",
    "truck": "rgb(255,140,0)",
    "person": "rgb(0,255,0)",
    "bus": "rgb(255,255,0)",
}
DEFAULT_COLOR = "rgb(0,128,255)"
INCIDENT_COLOR = "rgb(255,0,255)"


# =========================
# Data Processing
# =========================

def process_window_data(raw_data, location_filter="all"):
    """
    Process raw window data from backend into 1-second frames.
    
    Args:
        raw_data: Dict with vehicles, incidents, timestamp from backend
        location_filter: Filter by location ("all", "patterson", "foothill", etc.)
    
    Returns:
        Processed data dict with frames, or None if invalid
    """
    if not raw_data:
        return None
    
    try:
        vehicles = raw_data.get("vehicles", [])
        incidents = raw_data.get("incidents", [])
        timestamp = raw_data.get("timestamp")
        
        # Filter by location if needed
        if location_filter != "all":
            vehicles = [v for v in vehicles if v.get("location") == location_filter]
            incidents = [i for i in incidents if i.get("location") == location_filter]
        
        if not vehicles:
            return None
        
        # Sort all vehicles by timestamp
        vehicles_sorted = sorted(vehicles, key=lambda v: v.get("timestamp", ""))
        
        # Round timestamps to nearest second and group
        frames_dict = defaultdict(lambda: {"vehicles": [], "timestamp_display": None})
        
        for v in vehicles_sorted:
            ts_str = v.get("timestamp")
            if not ts_str:
                continue
            
            # Parse timestamp and round to second
            try:
                ts = pd.to_datetime(ts_str)
                ts_rounded = ts.floor('10ms')
                ts_key = ts_rounded.isoformat()
                
                frames_dict[ts_key]["vehicles"].append(v)
                if frames_dict[ts_key]["timestamp_display"] is None:
                    frames_dict[ts_key]["timestamp_display"] = ts_rounded.strftime('%H:%M:%S')
            except:
                continue
        
        # Sort by timestamp to create ordered frames
        sorted_timestamps = sorted(frames_dict.keys())
        
        frames = []
        for ts_key in sorted_timestamps:
            frame_data = frames_dict[ts_key]
            frames.append({
                "timestamp": ts_key,
                "timestamp_display": frame_data["timestamp_display"],
                "vehicles": frame_data["vehicles"]
            })
        
        return {
            "vehicles": vehicles_sorted,
            "incidents": incidents,
            "timestamp": timestamp,
            "frames": frames,
            "data_id": timestamp,  # Use timestamp as unique ID
            "location_filter": location_filter
        }
    
    except Exception as e:
        print(f"Error processing data: {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_center(vehicles, trim=0.1):
    pts = [(v.get("lat"), v.get("lon")) for v in vehicles]
    pts = [(lat, lon) for lat, lon in pts if lat is not None and lon is not None
           and math.isfinite(lat) and math.isfinite(lon)]
    if not pts:
        return dict(ANCHOR)

    lats = sorted(lat for lat, _ in pts)
    lons = sorted(lon for _, lon in pts)
    n = len(pts)
    k = int(n * trim)
    if n - 2 * k <= 0:
        k = 0

    lats2 = lats[k:n-k]
    lons2 = lons[k:n-k]
    return {"lat": sum(lats2) / len(lats2), "lon": sum(lons2) / len(lons2)}


def build_figure(center, frame_vehicles, all_vehicles, incidents, lat_off=0.0, lon_off=0.0):
    """Build map showing current frame with trajectory lines."""
    fig = go.Figure()
    
    # Draw trajectory lines (faint)
    trajectories = defaultdict(list)
    for v in all_vehicles:
        obj_id = str(v.get("id"))
        lat, lon = v.get("lat"), v.get("lon")
        if lat is not None and lon is not None:
            lat = lat + lat_off
            lon = lon + lon_off
        ts = v.get("timestamp")
        if lat is not None and lon is not None and ts:
            trajectories[obj_id].append((ts, lat, lon))
    
    for obj_id, points in trajectories.items():
        if len(points) < 2:
            continue
        
        points.sort()
        lats = [p[1] for p in points]
        lons = [p[2] for p in points]
        
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="lines",
            line=dict(width=1, color="rgba(150,150,255,0.3)"),
            hoverinfo="skip",
            showlegend=False,
        ))
    
    # Draw current frame vehicles
    if frame_vehicles:
        lons, lats, colors, texts, customdata = [], [], [], [], []
        
        for v in frame_vehicles:
            lat, lon = v.get("lat"), v.get("lon")
            if lat is None or lon is None:
                continue
            lat = lat + lat_off
            lon = lon + lon_off

            if not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            
            has_incident = len(v.get("incidents", [])) > 0
            
            if has_incident:
                color = INCIDENT_COLOR
                typ = f"{v.get('detected_type', 'unknown')} [INCIDENT]"
            else:
                typ = str(v.get("detected_type", "unknown")).lower()
                color = COLOR_MAP.get(typ, DEFAULT_COLOR)
            
            lats.append(lat)
            lons.append(lon)
            colors.append(color)
            texts.append(typ)
            
            speed = v.get("speed_mps", 0)
            accel = v.get("accel")
            
            customdata.append([
                v.get("id"),
                typ,
                f"{speed:.1f} m/s",
                f"{accel:.2f} m/s²" if accel is not None else "N/A"
            ])
        
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode="markers",
            marker=dict(size=7, opacity=0.95, color=colors),
            text=texts,
            customdata=customdata,
            hovertemplate=(
                "ID: %{customdata[0]}<br>"
                "Type: %{customdata[1]}<br>"
                "Speed: %{customdata[2]}<br>"
                "Accel: %{customdata[3]}<extra></extra>"
            ),
        ))
    
    # Add incident markers
    if incidents and frame_vehicles:
        inc_lats, inc_lons, inc_texts = [], [], []
        
        for inc in incidents:
            vehicle_ids = inc.get("vehicles", [])
            if not vehicle_ids:
                continue
            
            first_vehicle = next(
                (v for v in frame_vehicles if str(v.get("id")) == str(vehicle_ids[0])),
                None
            )
            
            if not first_vehicle:
                continue
            
            lat, lon = first_vehicle.get("lat"), first_vehicle.get("lon")
            if lat is None or lon is None:
                continue
            lat = lat + lat_off
            lon = lon + lon_off

            inc_lats.append(lat)
            inc_lons.append(lon)
            inc_texts.append(f"{inc['incident_type']}<br>Severity: {inc['severity']:.2f}")
        
        if inc_lats:
            fig.add_trace(go.Scattermapbox(
                lon=inc_lons,
                lat=inc_lats,
                mode="markers+text",
                marker=dict(size=22, opacity=0.7, color="rgb(255,0,0)"),
                text=["!"] * len(inc_lats),
                hovertext=inc_texts,
                hoverinfo="text",
            ))
    
    fig.update_layout(
        mapbox=dict(
            accesstoken=MAPBOX_TOKEN,
            style=MAP_STYLE,
            center=dict(lat=center["lat"], lon=center["lon"]),
            zoom=DEFAULT_ZOOM,
            pitch=60,
            bearing=30,
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision="constant",
    )
    
    return fig


# =========================
# Dash App
# =========================

def main():
    # Start immediately with empty data (don't wait)
    print("Starting map viewer (waiting for data)...")
    
    # Initialize with empty data structure
    data = {
        "vehicles": [],
        "incidents": [],
        "timestamp": "Waiting...",
        "frames": [],
        "data_id": None,
        "location_filter": "all"
    }
    
    # Create initial empty figure
    center = ANCHOR
    initial_frame = []
    fig = build_figure(center, initial_frame, [], [])
    
    app = Dash(__name__)
    app.title = "Live Traffic Feed"
    
    app.layout = html.Div([
        # Header with gradient
        html.Div([
            html.Div([
                html.H2("LIVE TRAFFIC MONITORING", style={
                    "color": "white",
                    "margin": "0",
                    "fontSize": "28px",
                    "fontWeight": "bold",
                    "textShadow": "2px 2px 4px rgba(0,0,0,0.3)"
                }),
                html.P("Real-time incident detection", style={
                    "color": "rgba(255,255,255,0.9)",
                    "margin": "5px 0 0 0",
                    "fontSize": "14px"
                }),
            ], style={"flex": "1"}),
            
            # Location selector dropdown
            html.Div([
                html.Label("Location:", style={
                    "color": "white",
                    "marginRight": "10px",
                    "fontSize": "14px",
                    "fontWeight": "600"
                }),
                dcc.Dropdown(
                    id="location-selector",
                    options=[],  # Will be populated dynamically
                    value="all",
                    clearable=False,
                    style={
                        "width": "200px",
                        "display": "inline-block"
                    }
                ),
            ], style={"marginRight": "20px", "display": "flex", "alignItems": "center"}),
            
            html.A(
                html.Button("Home", style={
                    "fontSize": "16px",
                    "padding": "12px 24px",
                    "background": "white",
                    "color": "#667eea",
                    "border": "2px solid white",
                    "borderRadius": "8px",
                    "cursor": "pointer",
                    "fontWeight": "600",
                    "boxShadow": "0 4px 15px rgba(0,0,0,0.2)",
                    "transition": "all 0.3s ease",
                    "marginRight": "10px",
                }),
                href="http://127.0.0.1:8050",
            ),
            html.A(
                html.Button("View All Incidents", style={
                    "fontSize": "16px",
                    "padding": "12px 24px",
                    "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "cursor": "pointer",
                    "fontWeight": "600",
                    "boxShadow": "0 4px 15px rgba(0,0,0,0.2)",
                    "transition": "all 0.3s ease",
                    "marginRight": "10px",
                }),
                href="http://127.0.0.1:8051",
                target="_blank",
            ),
            html.A(
                html.Button("Traffic Heatmap", style={
                    "fontSize": "16px",
                    "padding": "12px 24px",
                    "background": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                    "color": "white",
                    "border": "none",
                    "borderRadius": "8px",
                    "cursor": "pointer",
                    "fontWeight": "600",
                    "boxShadow": "0 4px 15px rgba(0,0,0,0.2)",
                    "transition": "all 0.3s ease",
                }),
                href="http://127.0.0.1:8052",
                target="_blank",
            ),
        ], style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "space-between",
            "padding": "20px 30px",
            "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "borderRadius": "12px",
            "marginBottom": "20px",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.1)"
        }),
        
        # Stats bar
        html.Div(id="stats-bar", style={
            "fontSize": "16px",
            "marginBottom": "15px",
            "padding": "15px 20px",
            "background": "white",
            "borderRadius": "10px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
            "border": "1px solid #e0e0e0"
        }),
        
        # Map
        html.Div([
            dcc.Graph(id="map", figure=fig, style={"height": "100%", "width": "100%"}),
            
            # No Data Overlay
            html.Div(
                id="no-data-overlay",
                style={"display": "none"},  # Hidden by default
                children=[
                    html.Div([
                        html.H2("No Data Received", style={
                            "color": "#666",
                            "marginBottom": "10px",
                            "fontSize": "32px"
                        }),
                        html.P("Waiting for vehicle data from camera...", style={
                            "color": "#999",
                            "fontSize": "16px"
                        }),
                        html.Div("", id="waiting-spinner", style={
                            "width": "50px",
                            "height": "50px",
                            "border": "5px solid #f3f3f3",
                            "borderTop": "5px solid #667eea",
                            "borderRadius": "50%",
                            "animation": "spin 1s linear infinite",
                            "margin": "20px auto"
                        }),
                    ], style={
                        "background": "white",
                        "padding": "40px",
                        "borderRadius": "12px",
                        "boxShadow": "0 4px 20px rgba(0,0,0,0.1)",
                        "textAlign": "center"
                    })
                ]
            )
        ], style={
            "height": "700px",
            "borderRadius": "12px",
            "overflow": "hidden",
            "boxShadow": "0 4px 20px rgba(0,0,0,0.1)",
            "marginBottom": "20px",
            "position": "relative"  # For absolute positioning of overlay
        }),
        
        # Controls panel
        html.Div([
            html.Div([
                html.Div(id="time-display", style={
                    "fontSize": "20px",
                    "fontWeight": "600",
                    "marginBottom": "15px",
                    "color": "#333"
                }),
                
                html.Div([
                    html.Button("Pause", id="pause-btn", n_clicks=0, style={
                        "marginRight": "10px",
                        "padding": "10px 20px",
                        "fontSize": "15px",
                        "borderRadius": "6px",
                        "border": "none",
                        "background": "#f44336",
                        "color": "white",
                        "cursor": "pointer",
                        "fontWeight": "500"
                    }),
                    html.Button("Resume", id="play-btn", n_clicks=0, style={
                        "padding": "10px 20px",
                        "fontSize": "15px",
                        "borderRadius": "6px",
                        "border": "none",
                        "background": "#4CAF50",
                        "color": "white",
                        "cursor": "pointer",
                        "fontWeight": "500"
                    }),
                ], style={"marginBottom": "25px"}),
                
                # Camera controls in grid
                html.Div([
                    html.Div([
                        html.Label("Camera Pitch", style={"fontWeight": "600", "marginBottom": "8px", "display": "block", "color": "#555"}),
                        dcc.Slider(0, 85, step=10, value=60, id="pitch-slider", marks={0: "0°", 45: "45°", 85: "85°"}),
                    ], style={"marginBottom": "20px"}),
                    
                    html.Div([
                        html.Label("Camera Bearing", style={"fontWeight": "600", "marginBottom": "8px", "display": "block", "color": "#555"}),
                        dcc.Slider(0, 360, step=15, value=30, id="bearing-slider", marks={0: "N", 90: "E", 180: "S", 270: "W"}),
                    ], style={"marginBottom": "20px"}),
                    
                    html.Div([
                        html.Label("Latitude Offset", style={"fontWeight": "600", "marginBottom": "8px", "display": "block", "color": "#555"}),
                        dcc.Slider(
                            id="lat-offset-slider",
                            min=-0.001, max=0.001, step=0.00001, value=0.0,
                            marks={-0.001: "−", 0.0: "0", 0.001: "+"},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], style={"marginBottom": "20px"}),
                    
                    html.Div([
                        html.Label("Longitude Offset", style={"fontWeight": "600", "marginBottom": "8px", "display": "block", "color": "#555"}),
                        dcc.Slider(
                            id="lon-offset-slider",
                            min=-0.001, max=0.001, step=0.00001, value=0.0,
                            marks={-0.001: "−", 0.0: "0", 0.001: "+"},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                    ], style={"marginBottom": "10px"}),
                    
                    html.Div(id="offset-display", style={"fontSize": "13px", "color": "#777"}),
                ])
            ])
        ], style={
            "padding": "25px",
            "background": "white",
            "borderRadius": "12px",
            "boxShadow": "0 2px 10px rgba(0,0,0,0.05)",
            "border": "1px solid #e0e0e0"
        }),
        
        dcc.Interval(id="animation-interval", interval=FRAME_INTERVAL_MS, n_intervals=0),
        dcc.Interval(id="reload-interval", interval=1000, n_intervals=0),
        
        dcc.Store(id="data-store", data=data),
        dcc.Store(id="current-frame", data=0),
        dcc.Store(id="playing", data=True),
        dcc.Store(id="alert-active", data=False),
        dcc.Store(id="last-alerted-data-id", data=None),
        dcc.Store(id="current-incident-id", data=None),
        dcc.Store(id="raw-data-store", data=None),  # Store raw unfiltered data

        # Enhanced Incident modal with video
        html.Div(
            id="incident-modal",
            style={"display": "none"},
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "padding": "0px",
                        "borderRadius": "16px",
                        "width": "600px",
                        "maxWidth": "90vw",
                        "boxShadow": "0 20px 60px rgba(0,0,0,0.3)",
                        "overflow": "hidden"
                    },
                    children=[
                        # Header
                        html.Div([
                            html.H2("Incident Detected", style={"margin": "0", "color": "white", "fontSize": "24px"}),
                        ], style={
                            "padding": "20px 30px",
                            "background": "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
                        }),
                        
                        # Body
                        html.Div([
                            html.Div(id="incident-modal-text", style={"marginBottom": "20px"}),
                            
                            # Video preview container
                            html.Div(id="video-preview-container", style={"marginTop": "20px"}),
                            
                        ], style={"padding": "30px"}),
                        
                        # Footer with buttons
                        html.Div([
                            html.Button("Watch Video", id="watch-video-btn", n_clicks=0, style={
                                "padding": "12px 24px",
                                "fontSize": "16px",
                                "borderRadius": "8px",
                                "border": "none",
                                "background": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
                                "color": "white",
                                "cursor": "pointer",
                                "fontWeight": "600",
                                "marginRight": "10px",
                                "boxShadow": "0 4px 15px rgba(0,0,0,0.2)"
                            }),
                            html.Button("Dismiss", id="incident-ok-btn", n_clicks=0, style={
                                "padding": "12px 24px",
                                "fontSize": "16px",
                                "borderRadius": "8px",
                                "border": "2px solid #ddd",
                                "background": "white",
                                "color": "#666",
                                "cursor": "pointer",
                                "fontWeight": "600"
                            }),
                        ], style={
                            "padding": "20px 30px",
                            "borderTop": "1px solid #eee",
                            "display": "flex",
                            "justifyContent": "flex-end"
                        }),
                    ],
                )
            ],
        ),
        
        html.Div(
            f"{PLAYBACK_FPS} FPS Playback • http://{HOST}:{PORT}",
            style={"marginTop": "20px", "textAlign": "center", "color": "#999", "fontSize": "13px"}
        ),
    ], style={
        "padding": "30px",
        "background": "#f5f7fa",
        "minHeight": "100vh",
        "fontFamily": "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
    })
    
    # Add CSS for spinner animation
    app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    @app.callback(
        Output("no-data-overlay", "style"),
        Input("data-store", "data"),
    )
    def toggle_no_data_overlay(data):
        """Show overlay when no data, hide when data exists."""
        if not data or not data.get("vehicles"):
            # Show overlay
            return {
                "display": "flex",
                "position": "absolute",
                "top": 0,
                "left": 0,
                "width": "100%",
                "height": "100%",
                "backgroundColor": "rgba(245, 247, 250, 0.95)",
                "zIndex": 1000,
                "alignItems": "center",
                "justifyContent": "center",
                "borderRadius": "12px"
            }
        else:
            # Hide overlay
            return {"display": "none"}
    
    @app.callback(
        Output("raw-data-store", "data"),
        Input("reload-interval", "n_intervals"),
    )
    def fetch_raw_data(n):
        """Fetch raw unfiltered data from shared memory."""
        raw_data = get_latest_data()
        if raw_data is None:
            return dash.no_update
        return raw_data
    
    @app.callback(
        Output("location-selector", "options"),
        Output("location-selector", "value"),
        Input("raw-data-store", "data"),
        State("location-selector", "value"),
    )
    def populate_location_dropdown(raw_data, current_value):
        """Populate dropdown with available locations from data."""
        if not raw_data or not raw_data.get("vehicles"):
            return [], "all"
        
        # Get unique locations from data
        locations = sorted(set(v.get("location") for v in raw_data["vehicles"] if v.get("location")))
        
        options = [{"label": "All Locations", "value": "all"}]
        options.extend([{"label": loc.title(), "value": loc} for loc in locations])
        
        # Default to first location if only one exists
        default_value = current_value if current_value else ("all" if len(locations) > 1 else locations[0])
        
        return options, default_value
    
    @app.callback(
        Output("data-store", "data"),
        Output("current-frame", "data"),
        Input("raw-data-store", "data"),
        Input("location-selector", "value"),
        State("data-store", "data"),
    )
    def process_and_filter_data(raw_data, location_filter, current_data):
        """Process raw data with location filter."""
        new_data = process_window_data(raw_data, location_filter)
        
        if new_data is None:
            return dash.no_update, dash.no_update
        
        # Check if data changed
        if current_data and new_data["data_id"] == current_data.get("data_id") and new_data["location_filter"] == current_data.get("location_filter"):
            return dash.no_update, dash.no_update
        
        # New data or filter changed! Reset to frame 0
        print(f"New data! {len(new_data['frames'])} frames, {len(new_data['vehicles'])} observations (location: {location_filter})")
        return new_data, 0
    
    @app.callback(
        Output("stats-bar", "children"),
        Input("data-store", "data"),
    )
    def update_stats(data):
        """Update stats bar."""
        if not data or not data.get("vehicles"):
            return html.Div("Waiting for data...", style={"color": "#999"})
        
        unique_ids = len(set(v.get("id") for v in data["vehicles"]))
        incident_count = len(data.get("incidents", []))
        location = data.get("location_filter", "all")
        
        return html.Div([
            html.Span(data.get('timestamp', 'Unknown'), style={"marginRight": "30px", "fontWeight": "500"}),
            html.Span(f"{unique_ids} Vehicles", style={"marginRight": "30px", "fontWeight": "500"}),
            html.Span(f"{len(data.get('frames', []))} Frames", style={"marginRight": "30px", "fontWeight": "500"}),
            html.Span(f"Location: {location.upper()}", style={"marginRight": "30px", "fontWeight": "500"}),
            html.Span(
                f"{incident_count} Incidents",
                style={
                    "fontWeight": "bold",
                    "color": "#f5576c" if incident_count > 0 else "#4CAF50",
                    "padding": "6px 12px",
                    "borderRadius": "6px",
                    "background": "rgba(245, 87, 108, 0.1)" if incident_count > 0 else "rgba(76, 175, 80, 0.1)"
                }
            ),
        ])

    @app.callback(
        Output("incident-modal", "style"),
        Output("incident-modal-text", "children"),
        Output("alert-active", "data"),
        Output("playing", "data", allow_duplicate=True),
        Output("last-alerted-data-id", "data"),
        Output("current-incident-id", "data"),
        Input("data-store", "data"),
        State("last-alerted-data-id", "data"),
        State("alert-active", "data"),
        prevent_initial_call=True,
    )
    def maybe_show_incident_modal(data, last_alerted_id, alert_active):
        if not data:
            raise PreventUpdate

        data_id = data.get("data_id")
        incidents = data.get("incidents") or []

        if not incidents:
            raise PreventUpdate

        if last_alerted_id is not None and data_id == last_alerted_id:
            raise PreventUpdate

        if alert_active:
            raise PreventUpdate

        # Build incident details
        lines = []
        for i, inc in enumerate(incidents[:5], 1):
            itype = inc.get("incident_type", "incident").upper()
            sev = inc.get("severity")
            sev_txt = f"{sev:.2f}" if isinstance(sev, (int, float)) else "N/A"
            vids = inc.get("vehicles", [])
            lines.append(
                html.Div([
                    html.Span(f"#{i} ", style={"fontWeight": "bold", "color": "#667eea"}),
                    html.Span(f"{itype}", style={"fontWeight": "600", "marginRight": "10px"}),
                    html.Span(f"Severity: {sev_txt}", style={"color": "#f5576c", "fontWeight": "600", "marginRight": "10px"}),
                    html.Span(f"Vehicles: {len(vids)}", style={"color": "#666"}),
                ], style={"marginBottom": "8px", "padding": "10px", "background": "#f8f9fa", "borderRadius": "6px"})
            )

        modal_text = html.Div([
            html.Div(f"{len(incidents)} incident(s) detected in this 15-second window", style={
                "marginBottom": "15px",
                "fontSize": "16px",
                "fontWeight": "600",
                "color": "#333"
            }),
            html.Div(lines),
        ])

        modal_style = {
            "display": "flex",
            "position": "fixed",
            "top": 0, "left": 0,
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": "rgba(0,0,0,0.7)",
            "zIndex": 9999,
            "alignItems": "center",
            "justifyContent": "center",
            "backdropFilter": "blur(4px)"
        }

        # Store first incident ID for video lookup
        first_incident_id = str(incidents[0].get("_id")) if incidents else None

        return modal_style, modal_text, True, False, data_id, first_incident_id
    
    @app.callback(
        Output("video-preview-container", "children"),
        Input("watch-video-btn", "n_clicks"),
        State("current-incident-id", "data"),
        prevent_initial_call=True,
    )
    def show_video_preview(n_clicks, incident_id):
        if not n_clicks or not incident_id:
            raise PreventUpdate
        
        # Fetch video for this incident
        try:
            response = requests.get(f"{API_BASE_URL}/videos/incident/{incident_id}", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    video_info = data.get("video")
                    video_id = video_info.get("_id")
                    
                    return html.Div([
                        html.Hr(style={"margin": "20px 0"}),
                        html.H4("Incident Video", style={"marginBottom": "15px"}),
                        html.Video(
                            src=f"{API_BASE_URL}/videos/{video_id}",
                            controls=True,
                            autoPlay=True,
                            style={
                                "width": "100%",
                                "borderRadius": "8px",
                                "boxShadow": "0 4px 15px rgba(0,0,0,0.2)"
                            }
                        ),
                    ])
                else:
                    return html.Div("No video available for this incident", style={"color": "#f5576c", "marginTop": "15px"})
            else:
                return html.Div("Could not load video", style={"color": "#f5576c", "marginTop": "15px"})
        except Exception as e:
            return html.Div(f"Error loading video: {str(e)}", style={"color": "#f5576c", "marginTop": "15px"})
    
    @app.callback(
        Output("incident-modal", "style", allow_duplicate=True),
        Output("alert-active", "data", allow_duplicate=True),
        Output("playing", "data", allow_duplicate=True),
        Output("video-preview-container", "children", allow_duplicate=True),
        Input("incident-ok-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def dismiss_incident_modal(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        hidden_style = {"display": "none"}
        return hidden_style, False, True, ""

    @app.callback(
        Output("time-display", "children"),
        Input("current-frame", "data"),
        State("data-store", "data"),
    )
    def update_time_display(frame_idx, data):
        """Show current frame timestamp."""
        if not data or not data.get("frames"):
            return "Waiting for data..."
        
        frames = data["frames"]
        if frame_idx >= len(frames):
            return "Loading..."
        
        frame = frames[frame_idx]
        ts_display = frame.get("timestamp_display", "")
        num_vehicles = len(frame.get("vehicles", []))
        
        return f"{ts_display} • Frame {frame_idx + 1}/{len(frames)} • {num_vehicles} vehicles"
    
    @app.callback(
        Output("current-frame", "data", allow_duplicate=True),
        Input("animation-interval", "n_intervals"),
        State("current-frame", "data"),
        State("data-store", "data"),
        State("playing", "data"),
        prevent_initial_call=True,
    )
    def advance_frame(n, current_frame, data, is_playing):
        """Advance to next frame every interval."""
        if not data or not data.get("frames") or not is_playing:
            return dash.no_update
        
        frames = data["frames"]
        
        # Don't loop - stop at last frame
        if current_frame >= len(frames) - 1:
            return dash.no_update  # Stay on last frame
        
        next_frame = current_frame + 1
        
        return next_frame
    
    @app.callback(
        Output("playing", "data"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_playback(play_clicks, pause_clicks):
        """Pause/resume playback."""
        ctx = dash.callback_context
        if not ctx.triggered:
            return True
        
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        return button_id == "play-btn"
    
    @app.callback(
        Output("map", "figure"),
        Input("current-frame", "data"),
        Input("pitch-slider", "value"),
        Input("bearing-slider", "value"),
        Input("lat-offset-slider", "value"),
        Input("lon-offset-slider", "value"),
        State("data-store", "data"),
    )
    def update_map(frame_idx, pitch, bearing, lat_offset, lon_offset, data):
        """Update map with current frame."""
        if not data or not data.get("frames"):
            # Return empty map centered on anchor
            empty_fig = build_figure(ANCHOR, [], [], [], lat_off=lat_offset, lon_off=lon_offset)
            empty_fig.update_layout(mapbox=dict(pitch=int(pitch), bearing=int(bearing)))
            return empty_fig
        
        frames = data["frames"]
        if frame_idx >= len(frames):
            return fig
        
        frame_vehicles = frames[frame_idx]["vehicles"]
        center = compute_center(data["vehicles"])
        
        new_fig = build_figure(center, frame_vehicles, data["vehicles"], data["incidents"], lat_off=lat_offset, lon_off=lon_offset)
        new_fig.update_layout(mapbox=dict(pitch=int(pitch), bearing=int(bearing)))
        
        return new_fig
    
    @app.callback(
        Output("offset-display", "children"),
        Input("lat-offset-slider", "value"),
        Input("lon-offset-slider", "value"),
    )
    def show_offsets(lat_off, lon_off):
        return f"Current offsets: lat {lat_off:+.6f}°, lon {lon_off:+.6f}°"

    print("\n" + "=" * 60)
    print("Live Traffic Feed - Enhanced UI")
    print("=" * 60)
    print(f"Open: http://{HOST}:{PORT}")
    print(f"Playing at {PLAYBACK_FPS} FPS (real-time)")
    print("=" * 60 + "\n")
    
    app.run(debug=False, host=HOST, port=PORT, use_reloader=False)


if __name__ == "__main__":
    main()