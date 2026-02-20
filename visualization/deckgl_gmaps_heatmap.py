#!/usr/bin/env python3
"""
deckgl_gmaps_heatmap.py

Writes a single HTML file that:
- Uses Google Maps as a basemap
- Uses deck.gl HeatmapLayer
- Reads ALL configuration from URL query params (no CLI args needed)

Typical use:
  python deckgl_gmaps_heatmap.py
Then serve the folder and open:
  http://127.0.0.1:8081/deckgl_gmaps_heatmap.html?apiKey=...&data=...&loc=patterson&range=hour

Query params supported:
  apiKey=YOUR_GOOGLE_KEY                         (required)
  data=http://127.0.0.1:8000/stats/combined      (recommended) OR a local json path like ./car_heatmap_static.json

  # If data is your FastAPI endpoint /stats/combined, you can also pass:
  range=hour|6hours|12hours|day|week|month        (default: hour)
  limit=10000                                     (default: 10000)
  loc=patterson                                   (optional)

  # Map view:
  lat=34.44                                       (optional; if omitted, computed from points)
  lng=-119.80                                     (optional; if omitted, computed from points)
  zoom=19                                         (default: 19)
  tilt=0                                          (default: 0)
  heading=0                                       (default: 0)

  # Heatmap tuning:
  radius=35                                       (default: 35)
  intensity=1.0                                   (default: 1.0)
  threshold=0.03                                  (default: 0.03)
  maxPoints=150000                                (optional downsample)
"""

from pathlib import Path


HTML = r"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Google Maps + deck.gl Heatmap</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />

    <style>
      html, body, #map { height: 100%; margin: 0; }
      #info {
        position: absolute; top: 10px; left: 10px; z-index: 10;
        background: rgba(255,255,255,0.92); padding: 10px 12px; border-radius: 10px;
        font-family: Arial, sans-serif; font-size: 13px; max-width: 420px;
      }
      #info input { width: 100%; }
      #err { color: #b00020; white-space: pre-wrap; margin-top: 8px; }
      .row { margin-top: 6px; opacity: .85; }
      code { font-size: 12px; }
    </style>
  </head>

  <body>
    <div id="info">
      <div><b>Google Maps + deck.gl Heatmap</b></div>
      <div class="row">This page is configured entirely via URL query parameters.</div>
      <div class="row"><b>Example:</b></div>
      <div class="row">
        <code>?apiKey=YOUR_KEY&data=http://127.0.0.1:8000/stats/combined&range=hour&loc=patterson&limit=20000</code>
      </div>
      <div class="row" id="status"></div>
      <div id="err"></div>
    </div>

    <div id="map"></div>

    <script type="module">
      import {GoogleMapsOverlay} from "https://unpkg.com/@deck.gl/google-maps@latest/dist/esm/index.js";
      import {HeatmapLayer} from "https://unpkg.com/@deck.gl/aggregation-layers@latest/dist/esm/index.js";

      function q() {
        return new URLSearchParams(window.location.search);
      }

      function qNum(params, key, defVal) {
        const v = params.get(key);
        if (v === null || v === "") return defVal;
        const n = Number(v);
        return Number.isFinite(n) ? n : defVal;
      }

      function setStatus(msg) {
        document.getElementById("status").textContent = msg;
      }

      function setErr(msg) {
        document.getElementById("err").textContent = msg || "";
      }

      function normalizePoints(rows) {
        // Supports:
        // 1) [{lat, lon/lng, weight/count}]
        // 2) response shape: {data: [...]} from your /stats/combined
        const arr = Array.isArray(rows) ? rows : (rows && Array.isArray(rows.data) ? rows.data : []);
        const pts = [];
        for (const r of arr) {
          const lat = Number(r.lat);
          const lng = Number(r.lng ?? r.lon);
          const w = Number(r.weight ?? r.count ?? 1);
          if (!Number.isFinite(lat) || !Number.isFinite(lng)) continue;
          pts.push({lat, lng, weight: Number.isFinite(w) ? w : 1});
        }
        return pts;
      }

      function reservoirSample(items, k) {
        if (!k || k <= 0 || items.length <= k) return items;
        const res = [];
        let n = 0;
        for (const it of items) {
          n++;
          if (res.length < k) res.push(it);
          else {
            const j = Math.floor(Math.random() * n);
            if (j < k) res[j] = it;
          }
        }
        return res;
      }

      function meanCenter(pts) {
        let sLat = 0, sLng = 0;
        for (const p of pts) { sLat += p.lat; sLng += p.lng; }
        return {lat: sLat / pts.length, lng: sLng / pts.length};
      }

      async function init() {
        const params = q();

        const apiKey = params.get("apiKey");
        if (!apiKey) {
          setErr("Missing required query param: apiKey");
          return;
        }

        // Dynamically load Google Maps JS using apiKey from query string
        await new Promise((resolve, reject) => {
          const s = document.createElement("script");
          s.src = `https://maps.googleapis.com/maps/api/js?key=${encodeURIComponent(apiKey)}`;
          s.async = true;
          s.defer = true;
          s.onload = resolve;
          s.onerror = () => reject(new Error("Failed to load Google Maps JS"));
          document.head.appendChild(s);
        });

        const dataBase = params.get("data");
        if (!dataBase) {
          setErr("Missing required query param: data (URL to JSON or /stats/combined endpoint)");
          return;
        }

        // If data points to your /stats/combined endpoint, we can append range/limit/loc
        const range = params.get("range") ?? "hour";
        const limit = params.get("limit") ?? "10000";
        const loc = params.get("loc");

        let dataUrl = dataBase;

        // Heuristic: if it looks like the combined stats endpoint, add its query params
        if (dataBase.includes("/stats/combined")) {
          const u = new URL(dataBase, window.location.href);
          u.searchParams.set("time_range", range);
          u.searchParams.set("limit", limit);
          if (loc) u.searchParams.set("location", loc);
          dataUrl = u.toString();
        }

        setStatus(`Loading data from: ${dataUrl}`);

        const resp = await fetch(dataUrl);
        if (!resp.ok) {
          const t = await resp.text().catch(() => "");
          throw new Error(`Data fetch failed: HTTP ${resp.status}\n\n${t}`);
        }
        const rows = await resp.json();

        let points = normalizePoints(rows);

        const maxPoints = qNum(params, "maxPoints", 0);
        points = reservoirSample(points, maxPoints);

        if (!points.length) {
          throw new Error("No valid points (need lat and lon/lng in data).");
        }

        const zoom = qNum(params, "zoom", 19);
        const tilt = qNum(params, "tilt", 0);
        const heading = qNum(params, "heading", 0);

        const radius = qNum(params, "radius", 35);
        const intensity = qNum(params, "intensity", 1.0);
        const threshold = qNum(params, "threshold", 0.03);

        const latOverride = params.get("lat");
        const lngOverride = params.get("lng");

        const center = (latOverride && lngOverride)
          ? { lat: Number(latOverride), lng: Number(lngOverride) }
          : meanCenter(points);

        const map = new google.maps.Map(document.getElementById("map"), {
          center,
          zoom,
          mapTypeId: "roadmap",
          tilt,
          heading,
        });

        const overlay = new GoogleMapsOverlay({
          layers: [
            new HeatmapLayer({
              id: "heat",
              data: points,
              getPosition: d => [d.lng, d.lat],
              getWeight: d => d.weight,
              radiusPixels: radius,
              intensity,
              threshold,
            })
          ]
        });

        overlay.setMap(map);

        setStatus(`✅ Loaded ${points.length.toLocaleString()} points. Center: ${center.lat.toFixed(6)}, ${center.lng.toFixed(6)}`);
        setErr("");
      }

      init().catch(err => {
        console.error(err);
        setErr(String(err?.stack || err));
      });
    </script>
  </body>
</html>
"""


def main() -> None:
    out = Path("deckgl_gmaps_heatmap.html")
    out.write_text(HTML, encoding="utf-8")
    print("✅ Wrote:", out.resolve())
    print("Open it with query params, e.g.:")
    print("  http://127.0.0.1:8081/deckgl_gmaps_heatmap.html?apiKey=YOUR_KEY&data=http://127.0.0.1:8000/stats/combined&range=hour&loc=patterson&limit=20000")


if __name__ == "__main__":
    main()
