import os
import re
import json
from dotenv import load_dotenv
from pathlib import Path
import pydeck as pdk




# -----------------------------
# Configuration
# -----------------------------

# ✅ Paste your Mapbox token here
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")
if not MAPBOX_TOKEN:
    raise RuntimeError("❌ MAPBOX_ACCESS_TOKEN is missing. Please set it in your .env file.")


CENTER = {"lat": 35.283, "lon": -120.66}
OPEN_BROWSER = True
USE_TERRAIN = False  # Set True to overlay terrain
HTML_OUT = "deck_trips.html"
MAP_STYLE = "mapbox://styles/mapbox/dark-v11"
TRAIL_LENGTH = 20


# -----------------------------
# Data loading
# -----------------------------
def load_frames_from_json(file_path: Path):
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Trips data prep
# -----------------------------
def frames_to_trips(frames):
    tracks = {}
    for t, frame in enumerate(frames):
        for obj in frame:
            obj_id = str(obj.get("id", f"anon_{t}"))
            x = float(obj.get("x", 0.5))
            y = float(obj.get("y", 0.5))
            lat = CENTER["lat"] + (y - 0.5) * 0.001
            lon = CENTER["lon"] + (x - 0.5) * 0.001
            if obj_id not in tracks:
                tracks[obj_id] = {
                    "path": [],
                    "timestamps": [],
                    "type": str(obj.get("type", "unknown")).lower(),
                }
            tracks[obj_id]["path"].append([lon, lat])
            tracks[obj_id]["timestamps"].append(t)

    color_map = {
        "car": [255, 0, 0],
        "truck": [255, 140, 0],
        "person": [0, 255, 0],
        "bus": [255, 255, 0],
    }

    trips = []
    max_t = 0
    for tr in tracks.values():
        trips.append(
            {
                "path": tr["path"],
                "timestamps": tr["timestamps"],
                "type": tr["type"],
                "color": color_map.get(tr["type"], [0, 128, 255]),
            }
        )
        if tr["timestamps"]:
            max_t = max(max_t, tr["timestamps"][-1])
    return trips, max_t


# -----------------------------
# Build deck
# -----------------------------
def build_deck(trip_data, current_time=0):
    pdk.settings.mapbox_api_key = MAPBOX_TOKEN

    trips_layer = pdk.Layer(
        "TripsLayer",
        id="trips",
        data=trip_data,
        get_path="path",
        get_timestamps="timestamps",
        get_color="color",
        opacity=0.85,
        width_min_pixels=3,
        rounded=True,
        trail_length=TRAIL_LENGTH,
        current_time=current_time,
    )

    layers = [trips_layer]

    if USE_TERRAIN:
        terrain_layer = pdk.Layer(
            "TerrainLayer",
            data=f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{{z}}/{{x}}/{{y}}.pngraw?access_token={MAPBOX_TOKEN}",
            elevation_decoder={"rScaler": 6553.6, "gScaler": 25.6, "bScaler": 0.1, "offset": -10000},
            texture=f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
            max_zoom=14,
            strategy="no-overlap",
            pickable=False,
        )
        layers.insert(0, terrain_layer)

    view_state = pdk.ViewState(
        latitude=CENTER["lat"],
        longitude=CENTER["lon"],
        zoom=17,
        pitch=60,
        bearing=0,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        views=[pdk.View("MapView", controller=True)],
        map_provider="mapbox",
        map_style=MAP_STYLE,
        tooltip={"text": "Type: {type}"},
        api_keys={"mapbox": MAPBOX_TOKEN},
    )
    return deck


# -----------------------------
# Main
# -----------------------------
def main():
    out_dir = Path("outputs")
    candidates = sorted(out_dir.glob("output*.json"))
    if not candidates:
        raise FileNotFoundError(f"No input found in {out_dir}/output*.json")

    latest = candidates[-1]
    frames = load_frames_from_json(latest)
    print(f"Loaded {len(frames)} frames from {latest.name}")

    trip_data, max_t = frames_to_trips(frames)
    print(f"Built {len(trip_data)} trip paths; max time index = {max_t}")

    deck = build_deck(trip_data, current_time=0)
    deck.to_html(HTML_OUT, open_browser=OPEN_BROWSER, notebook_display=False)
    print(f"✅ Created 3D trips map at {HTML_OUT}")


if __name__ == "__main__":
    main()
