#!/usr/bin/env python3

import os
import math
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go

import numpy as np
import pickle



# -----------------------------
# Config
# -----------------------------
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")  # Mapbox token from .env
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_ACCESS_TOKEN is missing in .env")

# Only used as a last-resort fallback if we cannot compute a center
ANCHOR = {"lat": 35.294099, "lon": -120.668143}

MAP_STYLE    = "mapbox://styles/mapbox/satellite-streets-v12"
DEFAULT_ZOOM = 18
INTERVAL_MS  = 100
MAX_ROWS     = 50000  # how many CSV rows to read at most

HOST = os.getenv("HOST", "127.0.0.1")
PORT = int(os.getenv("PORT", "8050"))

COLOR_MAP = {
    "car":    "rgb(255,0,0)",
    "truck":  "rgb(255,140,0)",
    "person": "rgb(0,255,0)",
    "bus":    "rgb(255,255,0)",
}
DEFAULT_COLOR = "rgb(0,128,255)"

MODEL_PATH = "bosh-metadata-reader/models/models_by_loc_2025-ll-21_10-29.pkl"
ANOMALY_THRESH = float(os.getenv("ANOMALY_THRESH", "0.5"))


# -----------------------------
# CSV loading & helpers
# -----------------------------
def _last4_hex(s: str) -> str:
    """Return last 4 hex digits of a string (robust to non-hex chars)."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s_hex = "".join(ch for ch in s if ch in "0123456789abcdef")
    return s_hex[-4:] if len(s_hex) >= 4 else s_hex

def load_frames_from_csv(file_path: Path, max_rows: int = MAX_ROWS):
    """
    Read CSV and return frames: list[ list[ {id_raw,id_short,type,speed,lat,lon} ] ],
    grouped by 'timestamp'. Only first `max_rows` rows are read.
    """
    df = pd.read_csv(file_path, nrows=max_rows)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Rename to expected
    if "detected_type" in df.columns:
        df.rename(columns={"detected_type": "type"}, inplace=True)
    if "object_id" in df.columns:
        df.rename(columns={"object_id": "id"}, inplace=True)
    if "speed_mps" in df.columns:
        df.rename(columns={"speed_mps": "speed"}, inplace=True)

    # Keep only needed columns
    keep = ["id", "timestamp", "type", "speed", "lat", "lon"]
    df = df[[c for c in keep if c in df.columns]]

    # Drop rows without lat/lon; coerce to numeric
    df = df.dropna(subset=["lat", "lon"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    if "speed" in df.columns:
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["lat", "lon"])

    # Parse timestamps and sort chronologically
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Build frames grouped by exact timestamp
    frames, timestamps = [], []
    for ts, group in df.groupby("timestamp", sort=True):
        objs = []
        for _, row in group.iterrows():
            raw_id = str(row["id"])
            objs.append({
                "id_raw": raw_id,
                "id_short": _last4_hex(raw_id),
                "type": str(row.get("type", "unknown")),
                "speed": float(row.get("speed", 0.0)),
                "lat": float(row["lat"]),
                "lon": float(row["lon"]),
                "time": float(row["timestamp"].value) if pd.notna(row["timestamp"]) else None,
            })
        frames.append(objs)
        timestamps.append(pd.Timestamp(ts))

    print(f"Loaded {len(frames)} frames from {len(df)} rows.")
    return frames, timestamps

def frame_to_points(frame):
    """
    Convert a frame (list of objs) to plotting arrays.
    Returns: lons, lats, colors, texts, customdata([id_short, type]).
    """
    lons, lats, colors, texts, customdata = [], [], [], [], []
    for obj in frame:
        try:
            lat = float(obj.get("lat"))
            lon = float(obj.get("lon"))
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue

        typ = str(obj.get("type", "unknown")).lower()
        short_id = str(obj.get("id_short") or "")[-4:]
        color = COLOR_MAP.get(typ, DEFAULT_COLOR)

        lats.append(lat); lons.append(lon); colors.append(color)
        texts.append(typ)
        customdata.append([short_id, typ])
    return lons, lats, colors, texts, customdata

def compute_center(frames):
    """Compute mean center from all object lat/lon in all frames."""
    lats, lons = [], []
    for f in frames:
        for o in f:
            lat, lon = o.get("lat"), o.get("lon")
            try:
                latf, lonf = float(lat), float(lon)
            except (TypeError, ValueError):
                continue
            if math.isfinite(latf) and math.isfinite(lonf):
                lats.append(latf); lons.append(lonf)
    if lats and lons:
        return {"lat": sum(lats)/len(lats), "lon": sum(lons)/len(lons)}
    return dict(ANCHOR)


# -----------------------------
# Plotly figure
# -----------------------------
def build_figure(center, lons, lats, colors, texts, customdata):
    fig = go.Figure()
    fig.add_trace(go.Scattermapbox(
        lon=lons,
        lat=lats,
        mode="markers",
        marker=dict(size=7, opacity=0.95, color=colors),
        text=texts,
        customdata=customdata,
        hovertemplate="ID: %{customdata[0]}<br>Type: %{customdata[1]}<extra></extra>",
        name="objects",
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
        uirevision="keep-view",  # preserves user camera while animating
    )
    return fig


# -----------------------------
# App
# -----------------------------
def main():
    # Find your CSVdef
    candidates = sorted(Path("").glob("combined_vehicle_stats_expandedNEW.csv"))
    if not candidates:
        raise FileNotFoundError("No 'combined_vehicle_stats_expandedNEW.csv' found in current directory")
    latest = candidates[-1]

    # Load frames
    frames, stamps = load_frames_from_csv(latest, max_rows=MAX_ROWS)
    n_frames = len(frames)
    if not n_frames:
        raise ValueError(f"{latest.name} contained no frames.")
    print(f"Loaded {n_frames} frames from {latest.name}")

    # Center camera from data
    center = compute_center(frames)
    print(f"Centered map at lat={center['lat']:.6f}, lon={center['lon']:.6f}")

    # Initial frame
    lons0, lats0, colors0, texts0, custom0 = frame_to_points(frames[0])
    print(f"Frame 0: {len(lats0)} points")
    fig = build_figure(center, lons0, lats0, colors0, texts0, custom0)

    app = Dash(__name__)
    app.title = "Mapbox Dots — Smooth Playback"
    app.playing = True

    app.layout = html.Div(
        ### implement the time frames in this part. need to display what time it is somewhere on the scren
        [
            html.H3("Frame-by-frame Dots (Plotly Scattermapbox)"),
            html.Div(
                id="time-label",
                style={"fontSize": "18px", "fontWeight": "600", "marginBottom": "8px"},
            ),
            dcc.Graph(id="map", figure=fig, style={"height": "640px", "width": "100%"}),
            dcc.Slider(
                0, max(n_frames - 1, 0),
                step=1,
                value=0,
                id="frame-slider",
                marks=None,
                tooltip={"always_visible": True},
            ),
            dcc.Interval(id="interval", interval=INTERVAL_MS, n_intervals=0),
            dcc.Store(id="play-offset", data=0),  # stores where to start playback
            html.Div(
                [
                    html.Button("▶️ Play", id="play-btn", n_clicks=0),
                    html.Button("⏸ Pause", id="pause-btn", n_clicks=0),
                ],
                style={"marginTop": "10px"},
            ),
            html.Div(f"Open http://{HOST}:{PORT}", style={"marginTop": "6px", "opacity": 0.7}),
        ],
        style={"padding": "16px"},
    )

    # Auto-advance the slider based on interval, honoring the stored offset
    @app.callback(
        Output("frame-slider", "value"),
        Input("interval", "n_intervals"),
        State("play-offset", "data"),
        prevent_initial_call=True,
    )
    
    def auto_advance(n, offset):
        if app.playing and n_frames:
            return (int(offset) + int(n)) % n_frames
        return dash.no_update

    # Update the map figure when the frame changes (via slider or auto-advance)
    @app.callback(
            Output("map", "figure"),
            Input("frame-slider", "value"),
            prevent_initial_call=True,
    )
    
    def update_frame(idx):
        idx = int(idx)
        lons, lats, colors, texts, cdata = frame_to_points(frames[idx])
        fig.update_traces(
            selector=dict(name="objects"),
            lon=lons, lat=lats,
            marker=dict(size=7, opacity=0.95, color=colors),
            text=texts,
            customdata=cdata,
            hovertemplate="ID: %{customdata[0]}<br>Type: %{customdata[1]}<extra></extra>",
        )
        return fig
    
    @app.callback(
        Output("time-label", "children"),
        Input("frame-slider", "value"),
    )
    def update_time_label(idx):
        idx = int(idx)
        if idx < 0 or idx >= len(stamps):
            return "Time: --"
        ts = stamps[idx]
        return f"Time: {ts.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}"

    # Play/Pause: on Play, set offset to current slider value and reset the interval counter
    @app.callback(
        Output("interval", "disabled"),
        Output("interval", "n_intervals"),
        Output("play-offset", "data"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        State("frame-slider", "value"),
        prevent_initial_call=True,
    )
    
    def toggle_play_pause(play_clicks, pause_clicks, slider_val):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update, dash.no_update, dash.no_update
        which = ctx.triggered[0]["prop_id"].split(".")[0]
        if which == "play-btn":
            app.playing = True
            # Resume from the frame the user slid to
            return False, 0, int(slider_val)  # enable interval, reset counter, set offset
        if which == "pause-btn":
            app.playing = False
            return True, dash.no_update, dash.no_update  # disable interval, keep counters
        return dash.no_update, dash.no_update, dash.no_update

    print(f"Running on: http://{HOST}:{PORT}")
    app.run(debug=True, host=HOST, port=PORT)


if __name__ == "__main__":
    main()


# Clear stagnant cars from parking lot? if time on screen < threshold (arbitrary) for N frames, remove
# gets rid of watching the cars in the parking lot etc. Would have to determine a method given long traffic light times
# perhaps based on speed and time on screen
# 
# Add some sort of time frame. Read the CSV entirely for its time frames and create an active one? 
# Given Frames jump from time to time based on objectID, strip the CSV just for the time frames, organize it, and create one using that??
#
# Add anomaly detection coloring based on pre-trained model !!!! Important to begin implementing the ML model into Visuals.
# FFMPEG is getting fixed but will still require time. 
# frame pausing in instance of anomaly? Needs to be tested from the DB side first, but could be useful to pause on anomalies for better viewing.
#
# 
#
#
#
#
#
#

