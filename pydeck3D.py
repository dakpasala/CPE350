import pydeck as pdk
import pandas as pd
import json
import time
from pathlib import Path

# ---------------------------------------------
# Load frames (your parsed traffic metadata)
# ---------------------------------------------
# with open("outputs/output3.json", "r") as f:
#     frames = json.load(f)

# Map (Foothill Blvd & Santa Rosa St for example)
CENTER = {"lat": 35.283, "lon": -120.66}

def load_frames_from_json(file_path: Path):
    """Load parsed Bosch metadata JSON."""
    with open(file_path, "r", encoding="utf-8") as f:
        frames = json.load(f)
    return frames

# ---------------------------------------------
# Function to create deck.gl layer from one frame
# ---------------------------------------------
# def make_layer_from_frame(frame):
#     df_rows = []
#     for obj in frame:
#         # Flexible coordinate support
#         if "lat" in obj and "lon" in obj:
#             lat, lon = obj["lat"], obj["lon"]
#         elif "x" in obj and "y" in obj:
#             # Convert normalized x/y to approximate GPS offset
#             lat = CENTER["lat"] + obj["y"] * 0.001
#             lon = CENTER["lon"] + obj["x"] * 0.001
#         else:
#             if lat is None or lon is None:
#             # Skip objects with no coordinates
#                 continue

#         df_rows.append({
#             "lat": float(lat) if lat not in [None, ""] else CENTER["lat"],
#             "lon": float(lon) if lon not in [None, ""] else CENTER["lon"],
#             "type": obj.get("type", "unknown")
#         })

#     df = pd.DataFrame(df_rows)

#     color_map = {
#         "car": [255, 0, 0],
#         "person": [0, 255, 0],
#         "truck": [255, 165, 0],
#         "bus": [255, 255, 0],
#     }

#     df["color"] = df["type"].map(color_map)
#     df["color"] = df["color"].apply(lambda c: c if isinstance(c, list) else [0, 0, 255])


#     return pdk.Layer(
#         "ScatterplotLayer",
#         df,
#         get_position=["lon", "lat"],
#         get_color="color",
#         get_radius=10,
#         pickable=True,
#         auto_highlight=True
#     )

# def make_animated_layer(frames):
#     """Flatten frames into a time-sequenced DataFrame for PyDeck animation."""
#     records = []
#     for t, frame in enumerate(frames):
#         for obj in frame:
#             x = obj.get("x", 0.5)
#             y = obj.get("y", 0.5)
#             lat = CENTER["lat"] + (y - 0.5) * 0.001
#             lon = CENTER["lon"] + (x - 0.5) * 0.001
#             obj_type = obj.get("type", "unknown").lower()
#             color = {
#                 "car": [255, 0, 0],
#                 "truck": [255, 140, 0],
#                 "person": [0, 255, 0],
#                 "bus": [255, 255, 0],
#             }.get(obj_type, [0, 128, 255])
#             records.append({
#                 "lat": lat,
#                 "lon": lon,
#                 "time": t,        # frame index as time
#                 "color": color,
#                 "type": obj_type
#             })

#     df = pd.DataFrame(records)

#     layer = pdk.Layer(
#         "ScatterplotLayer",
#         df,
#         get_position=["lon", "lat"],
#         get_fill_color="color",
#         get_radius=15,
#         pickable=True,
#         auto_highlight=True,
#     )

#     # Define view and animation parameters
#     view_state = pdk.ViewState(
#         latitude=CENTER["lat"], longitude=CENTER["lon"], zoom=17, pitch=45
#     )

#     # Add an animation description so Deck.GL interpolates frames over time
#     tooltip = {"text": "Type: {type}\nFrame: {time}"}

#     deck = pdk.Deck(
#         layers=[layer],
#         initial_view_state=view_state,
#         tooltip=tooltip,
#         map_style="mapbox://styles/mapbox/dark-v11",
#     )

#     # Add built-in JS animation (Deck.GL animation loop)
#     html = deck.to_html("deck_live.html", open_browser=True, notebook_display=False)
#     print("✅ Saved animated deck to deck_live.html")

#     return df

def make_trips_layer(frames):
    """Convert frame-by-frame detections into smooth animated 'trips'."""
    # Build dictionary of tracks (one per object_id)
    tracks = {}
    for t, frame in enumerate(frames):
        for obj in frame:
            obj_id = obj.get("id", f"anon_{t}_{id(obj)}")
            x = obj.get("x", 0.5)
            y = obj.get("y", 0.5)
            lat = CENTER["lat"] + (y - 0.5) * 0.001
            lon = CENTER["lon"] + (x - 0.5) * 0.001
            if obj_id not in tracks:
                tracks[obj_id] = {
                    "path": [],
                    "timestamps": [],
                    "type": obj.get("type", "unknown").lower(),
                }
            tracks[obj_id]["path"].append([lon, lat])
            tracks[obj_id]["timestamps"].append(t)

    # Prepare list for Deck.GL
    trip_data = []
    color_map = {
        "car": [255, 0, 0],
        "truck": [255, 140, 0],
        "person": [0, 255, 0],
        "bus": [255, 255, 0],
    }

    for obj_id, track in tracks.items():
        obj_type = track["type"]
        trip_data.append({
            "path": track["path"],
            "timestamps": track["timestamps"],
            "type": obj_type,
            "color": color_map.get(obj_type, [0, 128, 255])
        })

    # Create TripsLayer
    trips_layer = pdk.Layer(
        "TripsLayer",
        data=trip_data,
        get_path="path",
        get_timestamps="timestamps",
        get_color="color",
        opacity=0.8,
        width_min_pixels=3,
        rounded=True,
        trail_length=20,
        current_time=0,
    )

    view_state = pdk.ViewState(
        latitude=CENTER["lat"],
        longitude=CENTER["lon"],
        zoom=17,
        pitch=60,
        bearing=0,
    )

    deck = pdk.Deck(
        layers=[trips_layer],
        initial_view_state=view_state,
        map_style="mapbox://styles/mapbox/dark-v11",
        tooltip={"text": "Type: {type}"},
    )

    # Export to HTML with animation controls
    html_path = "deck_trips.html"
    deck.to_html(html_path, open_browser=True)
    print(f"✅ Created animated 3D trips map at {html_path}")
    
# ---------------------------------------------
# Create base map and view
# ---------------------------------------------
# view_state = pdk.ViewState(
#     latitude=CENTER["lat"],
#     longitude=CENTER["lon"],
#     zoom=17,
#     pitch=45,
# )

# ---------------------------------------------
# Display frames interactively
# ---------------------------------------------
# for idx, frame in enumerate(frames):
#     layer = make_layer_from_frame(frame)
#     deck = pdk.Deck(layers=[layer], initial_view_state=view_state, map_style="mapbox://styles/mapbox/light-v10")
#     deck.show()
#     print(f"Showing frame {idx+1}/{len(frames)}")
#     time.sleep(1/24)  # simulate ~24 fps
    
if __name__ == "__main__":
    out_dir = Path("outputs")
    latest = sorted(out_dir.glob("output*.json"))[-1]
    frames = load_frames_from_json(latest)
    print(f"Loaded {len(frames)} frames.")
    make_trips_layer(frames)

