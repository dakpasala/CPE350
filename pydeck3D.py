# pydeck3D.py

import os
import re
import json
from pathlib import Path
import pydeck as pdk

# -----------------------------
# Configuration
# -----------------------------
MAPBOX_TOKEN = "pk.eyJ1IjoibWNtaWtlMjkiLCJhIjoiY21oOWcxYXl0MG56eDJqcHkxeDl2OWx3dSJ9.xbojrrnBboUk3zq8lxLIuw"
CENTER = {"lat": 35.283, "lon": -120.66}
OPEN_BROWSER = True
USE_TERRAIN = False
HTML_OUT = "deck_trips.html"

# (Harmless) also set token in pydeck env
os.environ["MAPBOX_API_KEY"] = MAPBOX_TOKEN
pdk.settings.mapbox_api_key = MAPBOX_TOKEN


# -----------------------------
# Data loading
# -----------------------------
def load_frames_from_json(file_path: Path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Trips data prep
# -----------------------------
def frames_to_trips(frames):
    tracks = {}
    for t, frame in enumerate(frames):
        for obj in frame:
            obj_id = obj.get("id", f"anon_{t}")
            x = float(obj.get("x", 0.5))
            y = float(obj.get("y", 0.5))
            lat = CENTER["lat"] + (y - 0.5) * 0.001
            lon = CENTER["lon"] + (x - 0.5) * 0.001
            if obj_id not in tracks:
                tracks[obj_id] = {"path": [], "timestamps": [], "type": obj.get("type", "unknown").lower()}
            tracks[obj_id]["path"].append([lon, lat])
            tracks[obj_id]["timestamps"].append(t)

    color_map = {"car": [255, 0, 0], "truck": [255, 140, 0], "person": [0, 255, 0], "bus": [255, 255, 0]}
    trips, max_t = [], 0
    for tr in tracks.values():
        obj_type = tr["type"]
        trips.append({
            "path": tr["path"],
            "timestamps": tr["timestamps"],
            "type": obj_type,
            "color": color_map.get(obj_type, [0, 128, 255]),
        })
        if tr["timestamps"]:
            max_t = max(max_t, tr["timestamps"][-1])
    return trips, max_t


# -----------------------------
# Build deck (style URL includes token)
# -----------------------------
def build_deck(trip_data, current_time=0):
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
        trail_length=20,
        current_time=current_time,
    )

    terrain_layer = pdk.Layer(
        "TerrainLayer",
        data=f"https://api.mapbox.com/v4/mapbox.terrain-rgb/{{z}}/{{x}}/{{y}}.pngraw?access_token={MAPBOX_TOKEN}",
        elevation_decoder={"rScaler": 6553.6, "gScaler": 25.6, "bScaler": 0.1, "offset": -10000},
        texture="https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/{z}/{x}/{y}?access_token=" + MAPBOX_TOKEN,
        max_zoom=14,
        strategy="no-overlap",
        pickable=False,
    )

    layers = [trips_layer]
    if USE_TERRAIN:
        layers.insert(0, terrain_layer)

    view_state = pdk.ViewState(
        latitude=CENTER["lat"], longitude=CENTER["lon"], zoom=17, pitch=60, bearing=0
    )

    # Put the token in the style URL (helps, but not sufficient alone)
    style_with_token = f"https://api.mapbox.com/styles/v1/mapbox/dark-v11?access_token={MAPBOX_TOKEN}"

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style=style_with_token,
        tooltip={"text": "Type: {type}"},
        views=[pdk.View("MapView", controller=True)],
    )
    return deck


# -----------------------------
# HTML post-processing
# -----------------------------
def inject_mapbox_css_js_and_token(html_path: str, token: str):
    """Ensure Mapbox GL CSS+JS load first and token is set in <head>."""
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    css_snippet = """
    <!-- ✅ Mapbox GL CSS -->
    <link href="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.css" rel="stylesheet" />
    """
    js_snippet = """
    <!-- ✅ Mapbox GL JS -->
    <script src="https://api.mapbox.com/mapbox-gl-js/v2.15.0/mapbox-gl.js"></script>
    """
    token_snippet = f"""
    <!-- ✅ Set Mapbox token ASAP -->
    <script>
      (function() {{
        function setTok() {{
          try {{
            if (typeof mapboxgl !== 'undefined') {{
              mapboxgl.accessToken = "{token}";
              return true;
            }}
          }} catch(e) {{}}
          return false;
        }}
        if (!setTok()) {{
          var iv = setInterval(function() {{
            if (setTok()) clearInterval(iv);
          }}, 10);
        }}
      }})();
    </script>
    """

    # Prepend right after <head> to guarantee early execution
    if "<head>" in html:
        insertion = ""
        if "mapbox-gl.css" not in html:
            insertion += css_snippet
        if "mapbox-gl.js" not in html:
            insertion += js_snippet
        if "mapboxgl.accessToken" not in html:
            insertion += token_snippet
        if insertion:
            html = html.replace("<head>", "<head>\n" + insertion, 1)

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def inject_deck_constructor_token(html_path: str, token: str):
    """
    Insert mapboxApiAccessToken / mapboxAccessToken directly into:
      new deck.DeckGL({ ... })
    as the very first properties to avoid race conditions. Works even if pydeck
    changes whitespace or inserts 'var deckgl = ' etc.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Match new deck.DeckGL({ ... })
    # Capture group 1 = opening 'new deck.DeckGL({'
    # We then inject our token props immediately after '{'
    pattern = r"(new\s+deck\.DeckGL\(\s*\{)"
    replacement = r'new deck.DeckGL({mapboxApiAccessToken:"' + token + r'",mapboxAccessToken:"' + token + r'",'
    if re.search(pattern, html):
        # Avoid double-injecting
        if "mapboxApiAccessToken" not in html and "mapboxAccessToken" not in html:
            html = re.sub(pattern, replacement, html, count=1)

    # Also cover the common "var deckgl = new deck.DeckGL({" prefix (already handled by the same regex)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def inject_animation_js(html_path, max_time, fps=30):
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    anim_js = f"""
<script>
(function () {{
  try {{ if (typeof deckgl !== 'undefined') window.deckgl = deckgl; }} catch (e) {{}}

  function getTripsLayer(layers) {{
    return layers.find(l => l && l.id && l.id === 'trips');
  }}

  function startAnimation() {{
    if (!window.deckgl) {{
      let tries = 0;
      const it = setInterval(() => {{
        if (window.deckgl) {{ clearInterval(it); startAnimation(); }}
        else if (++tries > 60) {{ clearInterval(it); }}
      }}, 100);
      return;
    }}

    let t = 0;
    const T = {max_time if max_time > 0 else 1};
    const dt = 1 / {fps};

    function tick() {{
      t = (t + dt);
      if (t > T) t = 0;

      const layers = window.deckgl.props.layers || [];
      const trips = getTripsLayer(layers);
      if (trips && trips.clone) {{
        const newTrips = trips.clone({{currentTime: t}});
        const newLayers = layers.map(l => (l && l.id === 'trips') ? newTrips : l);
        window.deckgl.setProps({{layers: newLayers}});
      }}
      requestAnimationFrame(tick);
    }}
    requestAnimationFrame(tick);
  }}
  startAnimation();
}})();
</script>
"""
    if "</body>" in html and anim_js not in html:
        html = html.replace("</body>", anim_js + "\n</body>")

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


# -----------------------------
# Main
# -----------------------------
def main():
    out_dir = Path("outputs")
    latest = sorted(out_dir.glob("output*.json"))[-1]
    frames = load_frames_from_json(latest)
    print(f"Loaded {len(frames)} frames from {latest.name}")

    trip_data, max_t = frames_to_trips(frames)
    print(f"Built {len(trip_data)} trip paths; max time index = {max_t}")

    deck = build_deck(trip_data, current_time=0)

    deck.to_html(
        HTML_OUT,
        open_browser=OPEN_BROWSER,
        notebook_display=False,
    )

    # Order matters: load JS+CSS + set token FIRST, then ensure constructor has token, then animate
    inject_mapbox_css_js_and_token(HTML_OUT, MAPBOX_TOKEN)  # CSS, JS, early token
    inject_deck_constructor_token(HTML_OUT, MAPBOX_TOKEN)   # token inside DeckGL({...})
    inject_animation_js(HTML_OUT, max_time=max_t, fps=30)   # animation
    print(f"✅ Created animated 3D trips map at {HTML_OUT}")


if __name__ == "__main__":
    main()
