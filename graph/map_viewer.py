#!/usr/bin/env python3
"""
map_viewer.py

Continuous live feed with 15-second animated playback.
Waits for first data, then auto-plays forever.
"""

import os
import math
import json
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

MAP_STYLE = "mapbox://styles/mapbox/satellite-streets-v12"
DEFAULT_ZOOM = 18

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8050"))

# Play through 15 seconds in ~15 seconds (1x speed)
PLAYBACK_FPS = 10
FRAME_INTERVAL_MS = int(1000 / PLAYBACK_FPS)

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

def process_window_data(raw_data):
    """
    Process raw window data from backend into 1-second frames.
    
    Args:
        raw_data: Dict with vehicles, incidents, timestamp from backend
    
    Returns:
        Processed data dict with frames, or None if invalid
    """
    if not raw_data:
        return None
    
    try:
        vehicles = raw_data.get("vehicles", [])
        incidents = raw_data.get("incidents", [])
        timestamp = raw_data.get("timestamp")
        
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
            "data_id": timestamp  # Use timestamp as unique ID
        }
    
    except Exception as e:
        print(f"⚠️  Error processing data: {e}")
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
                typ = f"{v.get('detected_type', 'unknown')} 🚨"
            else:
                typ = str(v.get("detected_type", "unknown")).lower()
                color = COLOR_MAP.get(typ, DEFAULT_COLOR)
            
            lats.append(lat)  # Slight offset for visibility
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
                text=["⚠️"] * len(inc_lats),
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
    # Wait for first data from shared memory
    print("⏳ Waiting for first data from server...")
    data = None
    while data is None:
        raw_data = get_latest_data()
        data = process_window_data(raw_data)
        if data is None:
            time.sleep(1)
    
    print(f"✅ First data received! {len(data['frames'])} frames from {len(data['vehicles'])} observations")
    
    center = compute_center(data["vehicles"])
    initial_frame = data["frames"][0]["vehicles"] if data["frames"] else []
    fig = build_figure(center, initial_frame, data["vehicles"], data["incidents"])
    
    app = Dash(__name__)
    app.title = "Live Traffic Feed"
    
    app.layout = html.Div([
        html.H3("🔴 LIVE TRAFFIC FEED (15-20s delay)"),
        
        html.Div(id="stats-bar", style={
            "fontSize": "16px",
            "marginBottom": "10px",
            "padding": "10px",
            "backgroundColor": "#f0f0f0",
            "borderRadius": "5px"
        }),
        
        dcc.Graph(id="map", figure=fig, style={"height": "700px", "width": "100%"}),
        
        html.Div([
            html.Div(id="time-display", style={
                "fontSize": "18px",
                "fontWeight": "bold",
                "marginBottom": "10px"
            }),
            
            html.Div([
                html.Button("⏸️ Pause", id="pause-btn", n_clicks=0, style={"marginRight": "10px"}),
                html.Button("▶️ Resume", id="play-btn", n_clicks=0),
            ]),
            
            html.Div("Camera Pitch", style={"marginTop": "20px"}),
            dcc.Slider(0, 85, step=10, value=60, id="pitch-slider"),
            
            html.Div("Camera Bearing", style={"marginTop": "10px"}),
            dcc.Slider(0, 360, step=15, value=30, id="bearing-slider"),

            html.Div("Latitude offset (degrees)", style={"marginTop": "20px"}),
            
            dcc.Slider(
                id="lat-offset-slider",
                min=-0.001, max=0.001, step=0.00001, value=0.0, marks = {-0.001: "-0.001", -0.0005: "-0.0005", 0.0: "0", 0.0005: "+0.0005", 0.001: "+0.001"},
                tooltip={"placement": "bottom", "always_visible": True}
            ),

            html.Div("Longitude offset (degrees)", style={"marginTop": "10px"}),
            dcc.Slider(
                id="lon-offset-slider",
                min=-0.001, max=0.001, step=0.00001, value=0.0, marks = {-0.001: "-0.001", -0.0005: "-0.0005", 0.0: "0", 0.0005: "+0.0005", 0.001: "+0.001"},
                tooltip={"placement": "bottom", "always_visible": True}
            ),

            html.Div(id="offset-display", style={"marginTop": "10px", "opacity": 0.8}),

        ], style={"marginTop": "20px"}),
        
        dcc.Interval(id="animation-interval", interval=FRAME_INTERVAL_MS, n_intervals=0),
        dcc.Interval(id="reload-interval", interval=1000, n_intervals=0),
        
        dcc.Store(id="data-store", data=data),
        dcc.Store(id="current-frame", data=0),
        dcc.Store(id="playing", data=True),

        # Add these stores (near your existing dcc.Store lines)
        dcc.Store(id="alert-active", data=False),
        dcc.Store(id="last-alerted-data-id", data=None),

        # Add this modal overlay near the end of layout (but still inside the main Div children list)
        html.Div(
            id="incident-modal",
            style={
                "display": "none",              # toggled by callback
                "position": "fixed",
                "top": 0, "left": 0,
                "width": "100vw",
                "height": "100vh",
                "backgroundColor": "rgba(0,0,0,0.6)",
                "zIndex": 9999,
                "alignItems": "center",
                "justifyContent": "center",
            },
            children=[
                html.Div(
                    style={
                        "backgroundColor": "white",
                        "padding": "24px",
                        "borderRadius": "12px",
                        "width": "480px",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.25)",
                        "textAlign": "center",
                    },
                    children=[
                        html.H2("🚨 Accident detected", style={"marginTop": 0}),
                        html.Div(id="incident-modal-text", style={"marginBottom": "18px"}),
                        html.Button("OK", id="incident-ok-btn", n_clicks=0, style={
                            "fontSize": "16px",
                            "padding": "10px 18px",
                            "cursor": "pointer",
                        }),
                    ],
                )
            ],
        ),

        
        html.Div(
            f"Playing at {PLAYBACK_FPS} FPS | http://{HOST}:{PORT}",
            style={"marginTop": "10px", "opacity": 0.7}
        ),
    ], style={"padding": "20px"})
    
    @app.callback(
        Output("data-store", "data"),
        Output("current-frame", "data"),
        Input("reload-interval", "n_intervals"),
        State("data-store", "data"),
    )
    def check_for_new_data(n, current_data):
        """Check shared memory for new data every second."""
        raw_data = get_latest_data()
        new_data = process_window_data(raw_data)
        
        if new_data is None:
            return dash.no_update, dash.no_update
        
        # Check if data changed (compare timestamp)
        if current_data and new_data["data_id"] == current_data.get("data_id"):
            return dash.no_update, dash.no_update
        
        # New data arrived! Reset to frame 0
        print(f"📥 New data! {len(new_data['frames'])} frames, {len(new_data['vehicles'])} observations")
        return new_data, 0
    
    @app.callback(
        Output("stats-bar", "children"),
        Input("data-store", "data"),
    )
    def update_stats(data):
        """Update stats bar."""
        if not data:
            return "Waiting for data..."
        
        unique_ids = len(set(v.get("id") for v in data["vehicles"]))
        
        return html.Div([
            html.Span(f"📅 {data['timestamp']}", style={"marginRight": "20px"}),
            html.Span(f"🚗 Vehicles: {unique_ids}", style={"marginRight": "20px"}),
            html.Span(f"🎬 Frames: {len(data['frames'])}", style={"marginRight": "20px"}),
            html.Span(
                f"🚨 Incidents: {len(data['incidents'])}",
                style={"color": "red" if data["incidents"] else "green", "fontWeight": "bold"}
            ),
        ])


    @app.callback(
        Output("incident-modal", "style"),
        Output("incident-modal-text", "children"),
        Output("alert-active", "data"),
        Output("playing", "data", allow_duplicate=True),
        Output("last-alerted-data-id", "data"),
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

        # If no incidents, do nothing (and don't auto-unpause)
        if not incidents:
            raise PreventUpdate

        # If we've already alerted for this window, do nothing
        if last_alerted_id is not None and data_id == last_alerted_id:
            raise PreventUpdate

        # If modal already active, don't re-trigger
        if alert_active:
            raise PreventUpdate

        # Build a short message
        lines = []
        for inc in incidents[:5]:  # cap to avoid huge modal
            itype = inc.get("incident_type", "incident")
            sev = inc.get("severity")
            sev_txt = f"{sev:.2f}" if isinstance(sev, (int, float)) else "N/A"
            vids = inc.get("vehicles", [])
            lines.append(f"• {itype} | severity {sev_txt} | vehicles: {vids}")

        modal_text = html.Div([
            html.Div(f"Incidents in this 15s window: {len(incidents)}", style={"marginBottom": "10px"}),
            html.Pre("\n".join(lines), style={
                "textAlign": "left",
                "whiteSpace": "pre-wrap",
                "backgroundColor": "#f6f6f6",
                "padding": "10px",
                "borderRadius": "8px",
                "maxHeight": "220px",
                "overflowY": "auto",
            }),
            html.Div("Playback is paused until you click OK.", style={"marginTop": "10px", "opacity": 0.8}),
        ])

        # Show modal + pause
        modal_style = {
            "display": "flex",
            "position": "fixed",
            "top": 0, "left": 0,
            "width": "100vw",
            "height": "100vh",
            "backgroundColor": "rgba(0,0,0,0.6)",
            "zIndex": 9999,
            "alignItems": "center",
            "justifyContent": "center",
        }

        return modal_style, modal_text, True, False, data_id
    
    @app.callback(
        Output("incident-modal", "style", allow_duplicate=True),
        Output("alert-active", "data", allow_duplicate=True),
        Output("playing", "data", allow_duplicate=True),
        Input("incident-ok-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def dismiss_incident_modal(n_clicks):
        if not n_clicks:
            raise PreventUpdate

        hidden_style = {"display": "none"}
        return hidden_style, False, True


    
    @app.callback(
        Output("time-display", "children"),
        Input("current-frame", "data"),
        State("data-store", "data"),
    )
    def update_time_display(frame_idx, data):
        """Show current frame timestamp."""
        if not data or not data.get("frames"):
            return "⏱️ Waiting..."
        
        frames = data["frames"]
        if frame_idx >= len(frames):
            return "⏱️ Loading..."
        
        frame = frames[frame_idx]
        ts_display = frame.get("timestamp_display", "")
        num_vehicles = len(frame.get("vehicles", []))
        
        return f"⏱️ {ts_display} | Frame {frame_idx + 1}/{len(frames)} | {num_vehicles} vehicles"
    
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
        next_frame = (current_frame + 1) % len(frames)
        
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
            return fig
        
        frames = data["frames"]
        if frame_idx >= len(frames):
            return fig
        
        frame_vehicles = frames[frame_idx]["vehicles"]
        center = compute_center(data["vehicles"])
        # center = {"lat": center["lat"] + lat_offset, "lon": center["lon"] + lon_offset}
        
        new_fig = build_figure(center, frame_vehicles, data["vehicles"], data["incidents"], lat_off=lat_offset, lon_off=lon_offset)
        new_fig.update_layout(mapbox=dict(pitch=int(pitch), bearing=int(bearing)))
        
        return new_fig
    
    @app.callback(
    Output("offset-display", "children"),
    Input("lat-offset-slider", "value"),
    Input("lon-offset-slider", "value"),
    )
    def show_offsets(lat_off, lon_off):
        return f"Offsets: lat {lat_off:+.6f}, lon {lon_off:+.6f}"

    
    print("\n" + "=" * 60)
    print("Live Traffic Feed - Continuous Playback")
    print("=" * 60)
    print(f"📍 Open: http://{HOST}:{PORT}")
    print(f"🎬 Playing at {PLAYBACK_FPS} FPS (real-time)")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host=HOST, port=PORT)


if __name__ == "__main__":
    main()