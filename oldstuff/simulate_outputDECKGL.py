#!/usr/bin/env python3
"""
simulate_output_dash_deckgl.py
Fully working 3D animation of detected objects using Dash + Pydeck.
Smooth playback, rotation, and map-style view (Foothill × Santa Rosa).
"""

import re
import time
import pandas as pd
from pathlib import Path
import pydeck as pdk
from dash import Dash, dcc, html
from dash.dependencies import Input, Output

# -------------------------------------------------------
# Parse frames
# -------------------------------------------------------
def parse_output_txt(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    frame_blocks = re.split(r"🕒 Frame #[0-9]+ \| Time:", text)
    frames = []

    for block in frame_blocks[1:]:
        objs = []
        for match in re.finditer(
            r"🧩 Object ID:\s*(?P<id>\d+).*?"
            r"Type:\s*(?P<type>[A-Za-z]+).*?"
            r"Bounding Box:\s*top=\s*(?P<top>[-\d.]+),\s*bottom=\s*(?P<bottom>[-\d.]+),\s*left=\s*(?P<left>[-\d.]+),\s*right=\s*(?P<right>[-\d.]+)",
            block,
            re.DOTALL,
        ):
            t = match.group("type")
            left, right = float(match.group("left")), float(match.group("right"))
            top, bottom = float(match.group("top")), float(match.group("bottom"))
            x = (left + right) / 2
            y = (top + bottom) / 2
            objs.append({"type": t, "x": x, "y": y, "z": 0})
        frames.append(objs)

    return frames


# -------------------------------------------------------
# Build a Pydeck Deck object for a single frame
# -------------------------------------------------------
def build_deck(frame):
    color_map = {
        "car": [255, 0, 0],
        "person": [0, 255, 0],
        "truck": [255, 165, 0],
        "bus": [255, 255, 0],
    }

    # static roads
    roads = pd.DataFrame([
        {"path": [[-0.6, 1.1], [1.2, -0.3]], "name": "Foothill Blvd"},
        {"path": [[-0.6, -0.3], [1.2, 1.1]], "name": "Santa Rosa St"},
    ])

    road_layer = pdk.Layer(
        "PathLayer",
        data=roads,
        get_path="path",
        get_color=[120, 120, 120],
        width_scale=30,
        width_min_pixels=5,
        opacity=0.4,
        coordinate_system=1,  # IDENTITY
    )

    df = pd.DataFrame(frame)
    df["color"] = df["type"].map(lambda t: color_map.get(t.lower(), [0, 0, 255]))

    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["x", "y", "z"],
        get_color="color",
        get_radius=0.015,
        pickable=True,
        coordinate_system=1,
    )

    view_state = pdk.ViewState(
        longitude=0, latitude=0, zoom=1.7, pitch=50, bearing=30
    )

    return pdk.Deck(
        layers=[road_layer, scatter_layer],
        initial_view_state=view_state,
        map_style=None,
        tooltip={"text": "{type}"},
    )


# -------------------------------------------------------
# Dash app for interactive playback
# -------------------------------------------------------
def run_dash_app(frames):
    app = Dash(__name__)

    # Initial HTML export for first frame
    initial_deck = build_deck(frames[0])
    initial_html = initial_deck.to_html(as_string=True, notebook_display=False)

    app.layout = html.Div(
        [
            html.H2("🚦 Foothill × Santa Rosa 3D Intersection (Deck.gl)"),
            html.Iframe(
                id="deck-frame",
                srcDoc=initial_html,
                style={"width": "100%", "height": "600px", "border": "none"},
            ),
            dcc.Slider(
                0, len(frames) - 1,
                step=1,
                value=0,
                id="frame-slider",
                marks=None,
                tooltip={"always_visible": True}
            ),
            dcc.Interval(id="interval", interval=100, n_intervals=0),
            html.Div(
                [
                    html.Button("▶️ Play", id="play-btn", n_clicks=0),
                    html.Button("⏸ Pause", id="pause-btn", n_clicks=0),
                ],
                style={"marginTop": "10px"},
            ),
        ],
        style={"padding": "20px"},
    )

    # State
    app.playing = True

    @app.callback(
        Output("frame-slider", "value"),
        Input("interval", "n_intervals"),
        prevent_initial_call=True,
    )
    def auto_advance(n):
        if app.playing:
            return (n % len(frames))
        return dash.no_update

    @app.callback(
        Output("deck-frame", "srcDoc"),
        Input("frame-slider", "value"),
    )
    def update_frame(idx):
        deck = build_deck(frames[idx])
        return deck.to_html(as_string=True, notebook_display=False)

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
        button = ctx.triggered[0]["prop_id"].split(".")[0]
        if button == "play-btn":
            app.playing = True
            return False
        elif button == "pause-btn":
            app.playing = False
            return True
        return dash.no_update

    app.run(debug=True)


# -------------------------------------------------------
# Run
# -------------------------------------------------------
def main():
    out_dir = Path("outputs")
    latest = sorted(out_dir.glob("output*.txt"))[-1]
    print(f"🎥 Visualizing from: {latest}")

    frames = parse_output_txt(latest)
    print(f"✅ Parsed {len(frames)} frames.")
    time.sleep(1)
    run_dash_app(frames)


if __name__ == "__main__":
    main()
