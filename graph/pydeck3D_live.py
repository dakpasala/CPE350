#!/usr/bin/env python3
# pydeck3D.py (JSON-capable)
# Keeps old CSV path commented out (for ref), adds JSON/JSONL loader.

import os
import math
from pathlib import Path
from dotenv import load_dotenv

import configparser
import pymongo

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
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_ACCESS_TOKEN is missing in .env")

ANCHOR = {"lat": 35.294099, "lon": -120.668143}

MAP_STYLE    = "mapbox://styles/mapbox/satellite-streets-v12"
DEFAULT_ZOOM = 18
INTERVAL_MS  = 100
MAX_ROWS     = 50000

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
# Helpers
# -----------------------------
def _last4_hex(s: str) -> str:
    """last 4 hex chars (robust)"""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip().lower()
    s_hex = "".join(ch for ch in s if ch in "0123456789abcdef")
    return s_hex[-4:] if len(s_hex) >= 4 else s_hex


def get_mongo_collection(ini_path: Path, db_name: str, coll_name: str):
    cfg = configparser.ConfigParser()
    cfg.read(str(ini_path))

    uri = cfg.get("DEFAULT", "database", fallback=None)
    if not uri:
        raise KeyError("connection.ini missing [DEFAULT] database = ...")

    client = pymongo.MongoClient(uri.strip())
    return client[db_name][coll_name]


def load_frames_from_mongo(ini_path: Path, db_name: str, coll_name: str,
                          limit: int = MAX_ROWS, sort_field: str = "timestamp"):
    coll = get_mongo_collection(ini_path, db_name, coll_name)

    proj = {
        "_id": 0,
        "object_id": 1,
        "timestamp": 1,
        "detected_type": 1,
        "speed_mps": 1,
        "accel": 1,
        "jerk": 1,
        "heading_deg": 1,
        "d_heading_deg": 1,
        "nn_dist_m": 1,
        "closing_rate_mps": 1,
        "ttc_s": 1,
        "rel_speed_mps": 1,
        "heading_diff_deg": 1,
        "zone_change": 1,
        "path_gap": 1,
        "certainty": 1,
        "is_confident": 1,
        "lat": 1,
        "lon": 1,
        "location": 1,
    }

    # pull newest first, then reverse to keep chronological order
    cur = coll.find({}, proj).sort(sort_field, pymongo.DESCENDING).limit(int(limit))
    rows = list(cur)
    rows.reverse()
    print(f"Mongo raw docs fetched: {len(rows)}")
    if rows:
        s = rows[0]
        print("Sample keys:", sorted(s.keys()))
        print("Sample timestamp:", s.get("timestamp"))
        print("Sample lat/lon:", s.get("lat"), s.get("lon"))
        print("Sample type/speed:", s.get("detected_type"), s.get("speed_mps"))
    if not rows:
        return [], []

    df = pd.DataFrame.from_records(rows)
    df.columns = [c.strip().lower() for c in df.columns]

    # normalize names to your existing pipeline
    if "detected_type" in df.columns:
        df.rename(columns={"detected_type": "type"}, inplace=True)
    if "object_id" in df.columns:
        df.rename(columns={"object_id": "id"}, inplace=True)
    if "speed_mps" in df.columns:
        df.rename(columns={"speed_mps": "speed"}, inplace=True)

    keep = ["id", "timestamp", "type", "speed", "lat", "lon"]
    df = df[[c for c in keep if c in df.columns]]

    df = df.dropna(subset=["lat", "lon"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["speed"] = pd.to_numeric(df.get("speed", 0.0), errors="coerce").fillna(0.0)
    df = df.dropna(subset=["lat", "lon"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    frames, stamps = [], []
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
        stamps.append(pd.Timestamp(ts))

    print(f"Loaded {len(frames)} frames from {len(df)} rows (Mongo).")
    return frames, stamps


def frame_to_points(frame):
    """frame(list objs) -> arrays for plotly"""
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

        lats.append(lat)
        lons.append(lon)
        colors.append(color)
        texts.append(typ)
        customdata.append([short_id, typ])

    return lons, lats, colors, texts, customdata


def compute_center(frames):
    """mean center from all points"""
    lats, lons = [], []
    for f in frames:
        for o in f:
            try:
                latf = float(o.get("lat"))
                lonf = float(o.get("lon"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(latf) and math.isfinite(lonf):
                lats.append(latf)
                lons.append(lonf)
    if lats and lons:
        return {"lat": sum(lats) / len(lats), "lon": sum(lons) / len(lons)}
    return dict(ANCHOR)


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
        uirevision=False,
    )
    return fig


def main():
    # -----------------------------
    # Mongo settings
    # -----------------------------
    INI_PATH = Path(__file__).resolve().parent / "connection.ini"

    # This is the DB name your friend's script uses:
    DB_NAME = "camera-counts"

    # IMPORTANT:
    # Set this to the collection that ALREADY has top-level lat/lon per row
    # (the "formatted JSON" collection your teammate mentioned).
    #
    # If you set this wrong (ex: the raw "vehicles" that has mapPath),
    # you will likely load 0 frames.
    COLL_NAME = os.getenv("MONGO_COLLECTION", "combined_stats")

    # How many rows to fetch max
    LIMIT = int(os.getenv("MONGO_LIMIT", str(MAX_ROWS)))

    print(f"Mongo: ini={INI_PATH} db={DB_NAME} coll={COLL_NAME} limit={LIMIT}")

    # -----------------------------
    # Load frames from Mongo
    # -----------------------------
    frames, stamps = load_frames_from_mongo(
        ini_path=INI_PATH,
        db_name=DB_NAME,
        coll_name=COLL_NAME,
        limit=LIMIT,
        sort_field="timestamp",
    )

    n_frames = len(frames)
    if not n_frames:
        raise ValueError(
            "No frames loaded from Mongo. Most likely COLL_NAME is wrong "
            "(you pointed at a collection that doesn't have top-level lat/lon)."
        )

    print(f"Loaded {n_frames} frames from Mongo")

    # -----------------------------
    # Center + initial fig
    # -----------------------------
    center = compute_center(frames)
    print(f"Centered map at lat={center['lat']:.6f}, lon={center['lon']:.6f}")

    lons0, lats0, colors0, texts0, custom0 = frame_to_points(frames[0])
    fig = build_figure(center, lons0, lats0, colors0, texts0, custom0)

    # -----------------------------
    # Dash app (same as before)
    # -----------------------------
    app = Dash(__name__)
    app.title = "Mapbox Dots — Smooth Playback"
    app.playing = True

    app.layout = html.Div(
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
            html.Div("Camera Pitch"),
            dcc.Slider(
                0, 85,
                step=1,
                value=60,
                id="pitch-slider",
            ),
            html.Div("Camera Bearing"),
            dcc.Slider(
                0, 360,
                step=1,
                value=30,
                id="bearing-slider",
            ),
            dcc.Interval(id="interval", interval=INTERVAL_MS, n_intervals=0),
            dcc.Store(id="play-offset", data=0),
            html.Div(
                [
                    html.Button("Play", id="play-btn", n_clicks=0),
                    html.Button("Pause", id="pause-btn", n_clicks=0),
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
        State("play-offset", "data"),
        prevent_initial_call=True,
    )
    def auto_advance(n, offset):
        if app.playing and n_frames:
            return (int(offset) + int(n)) % n_frames
        return dash.no_update

    @app.callback(
        Output("map", "figure"),
        Input("frame-slider", "value"),
        Input("pitch-slider", "value"),
        Input("bearing-slider", "value"),
        prevent_initial_call=True,
    )
    def update_frame(idx, pitch, bearing):
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
        fig.update_layout(
            mapbox=dict(
                pitch=int(pitch),
                bearing=int(bearing),
            )
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
            return False, 0, int(slider_val)
        if which == "pause-btn":
            app.playing = False
            return True, dash.no_update, dash.no_update
        return dash.no_update, dash.no_update, dash.no_update

    print(f"Running on: http://{HOST}:{PORT}")
    app.run(debug=True, host=HOST, port=PORT)

if __name__ == "__main__":
    main()