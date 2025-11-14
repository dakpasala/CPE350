#!/usr/bin/env python3

import os
import json
import math
from pathlib import Path
from dotenv import load_dotenv

import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# -----------------------------
# Config
# -----------------------------
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN") #load mapbox token from .env
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_ACCESS_TOKEN is missing in .env")

# Anchor (adjust to your site)
ANCHOR = {"lat": 35.294289, "lon": -120.668143} #where map is centered

MAP_STYLE = "mapbox://styles/mapbox/satellite-streets-v12" #can change
DEFAULT_ZOOM = 18
INTERVAL_MS = 100
HOST = os.getenv("HOST", "127.0.0.1") #local host
PORT = int(os.getenv("PORT", "8050"))

COLOR_MAP = { #color coded
    "car":    "rgb(255,0,0)",
    "truck":  "rgb(255,140,0)",
    "person": "rgb(0,255,0)",
    "bus":    "rgb(255,255,0)",
}
DEFAULT_COLOR = "rgb(0,128,255)"

# -----------------------------
# Data loading & helpers
# -----------------------------
def load_frames_from_json(file_path: Path): #load in json file
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

def obj_to_abs_latlon(obj): #calculate actual position based on center of graph
    """Return (lat, lon) in absolute degrees."""
    lat, lon = obj.get("lat"), obj.get("lon")
    if lat is not None and lon is not None:
        lat = float(lat); lon = float(lon)
        if abs(lat) < 1 and abs(lon) < 1:
            return ANCHOR["lat"] + lat, ANCHOR["lon"] + lon
        return lat, lon
    x = float(obj.get("x", 0.5)); y = float(obj.get("y", 0.5))
    return ANCHOR["lat"] + (y - 0.5) * 0.001, ANCHOR["lon"] + (x - 0.5) * 0.001

def frame_to_points(frame):
    """Return lists: lons, lats, colors (strings), texts."""
    lons, lats, colors, texts = [], [], [], []
    for obj in frame:
        lat, lon = obj_to_abs_latlon(obj)
        if not (math.isfinite(lat) and math.isfinite(lon)):
            continue
        typ = str(obj.get("type", "unknown")).lower()
        color = COLOR_MAP.get(typ, DEFAULT_COLOR)
        lats.append(lat); lons.append(lon); colors.append(color)
        texts.append(typ)
    return lons, lats, colors, texts

def compute_center(frames):
    lats, lons = [], []
    for f in frames:
        for o in f:
            lat, lon = obj_to_abs_latlon(o)
            if math.isfinite(lat) and math.isfinite(lon):
                lats.append(lat); lons.append(lon)
    if lats and lons:
        return {"lat": sum(lats)/len(lats), "lon": sum(lons)/len(lons)}
    return dict(ANCHOR)

# -----------------------------
# Build initial figure
# -----------------------------
def build_figure(center, lons, lats, colors, texts):
    fig = go.Figure()

    fig.add_trace(go.Scattermapbox(
        lon=lons,
        lat=lats,
        mode="markers",
        marker=dict(
            size=7,                # dot size in pixels
            opacity=0.95,
            color=colors,
        ),
        text=texts,
        hovertemplate="Type: %{text}<extra></extra>",
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
        uirevision = "keep-view",
    )
    return fig

# -----------------------------
# App
# -----------------------------
def main():
    out_dir = Path("")
    candidates = sorted(out_dir.glob("output3.json"))
    if not candidates:
        raise FileNotFoundError(f"No input found in {out_dir}/output*.json")

    latest = candidates[-1]
    frames = load_frames_from_json(latest)
    n_frames = len(frames)
    if not n_frames:
        raise ValueError(f"{latest.name} contained no frames.")
    print(f"Loaded {n_frames} frames from {latest.name}")

    center = compute_center(frames)
    print(f"Centered map at lat={center['lat']:.6f}, lon={center['lon']:.6f}")

    # initial frame
    lons0, lats0, colors0, texts0 = frame_to_points(frames[0])
    print(f"Frame 0: {len(lats0)} points")
    fig = build_figure(center, lons0, lats0, colors0, texts0)

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
        lons, lats, colors, texts = frame_to_points(frames[idx])
        # just update data in the existing figure
        fig.update_traces(
            selector=dict(name="objects"),
            lon=lons, lat=lats,
            marker=dict(size=7, opacity=0.95, color=colors),
            text=texts,
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