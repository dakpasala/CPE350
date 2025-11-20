#!/usr/bin/env python3

import os
import math
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go


# -----------------------------
# Config
# -----------------------------
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")  # Mapbox token from .env
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_ACCESS_TOKEN is missing in .env")

# Only used as a last-resort fallback if we cannot compute a center
ANCHOR = {"lat": 35.294099, "lon": -120.668143}

MAP_STYLE   = "mapbox://styles/mapbox/satellite-streets-v12"
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

    # Parse timestamps and sort chronologically (prevents “random dots” feel)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Build frames grouped by exact timestamp
    frames = []
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
            })
        frames.append(objs)

    print(f"✅ Loaded {len(frames)} frames from {len(df)} rows.")
    return frames

def frame_to_points(frame):
    """
    Convert a frame (list of objs) to plotting arrays.
    Returns: lons, lats, colors, texts, customdata([id_short, type, id_raw?]).
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
        texts.append(typ)                         # optional text label
        customdata.append([short_id, typ])       # hover shows short id + type
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
    # Find your CSV
    candidates = sorted(Path("").glob("combined_vehicle_stats_expandedNEW.csv"))
    if not candidates:
        raise FileNotFoundError("No 'combined_vehicle_stats_with_derivatives.csv' found in current directory")
    latest = candidates[-1]

    # Load frames
    frames = load_frames_from_csv(latest, max_rows=MAX_ROWS)
    n_frames = len(frames)
    if not n_frames:
        raise ValueError(f"{latest.name} contained no frames.")
    print(f"Loaded {n_frames} frames from {latest.name}")

    # Center camera from data
    center = compute_center(frames)
    print(f"📍 Centered map at lat={center['lat']:.6f}, lon={center['lon']:.6f}")

    # Initial frame
    lons0, lats0, colors0, texts0, custom0 = frame_to_points(frames[0])
    print(f"Frame 0: {len(lats0)} points")
    fig = build_figure(center, lons0, lats0, colors0, texts0, custom0)

    # Dash app
    app = Dash(__name__)
    app.title = "Mapbox Dots — Smooth Playback"
    app.playing = True

    app.layout = html.Div(
        [
            html.H3("Frame-by-frame Dots (Plotly Scattermapbox)"),
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

    @app.callback(
        Output("frame-slider", "value"),
        Input("interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def auto_advance(n):
        if app.playing and n_frames:
            return n % n_frames
        return dash.no_update

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
        Output("interval", "disabled"),
        Input("play-btn", "n_clicks"),
        Input("pause-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def toggle_play_pause(play_clicks, pause_clicks):
        ctx = dash.callback_context
        if not ctx.triggered:
            return dash.no_update
        which = ctx.triggered[0]["prop_id"].split(".")[0]
        if which == "play-btn":
            app.playing = True
            return False
        if which == "pause-btn":
            app.playing = False
            return True
        return dash.no_update

    print(f"Running on: http://{HOST}:{PORT}")
    app.run(debug=True, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
