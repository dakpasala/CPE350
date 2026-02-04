#!/usr/bin/env python3
"""
car_heatmapV2.py  (MongoDB → spatial density heatmap)

Goal:
- Produce a heatmap of "time spent / density" over an intersection using (lat, lon) samples.
- Supports static + animated heatmaps (time-binned).
- Supports either:
  A) Binning (rounded grid)  -> smaller output, but can look “grid-like”
  B) No-binning (raw points) -> smoother/continuous look (recommended for “accurate paths”)
     Uses reservoir sampling to keep memory bounded.

Key feature you asked for:
- You can shift the HEATMAP POINTS without moving the MAP CENTER:
  --points-lat-offset / --points-lon-offset shift the plotted points
  --center-lat-offset / --center-lon-offset shift only the map center

Examples:

# Static heatmap (no binning; smoother)
python car_heatmapV2.py --mongo-uri "mongodb+srv://..." --db camera-counts --collection combined-stats \
  --location patterson --start 2026-01-21 --end 2026-01-22 --no-binning --max-points 200000

# Animated heatmap in 5-minute bins (no binning; per-frame sampling)
python car_heatmapV2.py --mongo-uri "mongodb+srv://..." --db camera-counts --collection combined-stats \
  --location patterson --start 2026-01-21 --end 2026-01-22 --interval 5min --no-binning \
  --max-points-per-frame 30000

# Shift heatmap points slightly west (negative lon) but keep map center unchanged
python car_heatmapV2.py --mongo-uri "mongodb+srv://..." --db camera-counts --collection combined-stats \
  --location patterson --start 2026-01-21 --end 2026-01-22 --no-binning \
  --points-lon-offset -0.00010

# Angle the map view (Mapbox styles)
python car_heatmapV2.py --mongo-uri "mongodb+srv://..." --db camera-counts --collection combined-stats \
  --location patterson --start 2026-01-21 --end 2026-01-22 --no-binning \
  --pitch 55 --bearing -20
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple
import math

import pandas as pd
import plotly.express as px

try:
    import pymongo
except Exception:
    pymongo = None


# -----------------------------
# .env loading (walk up parents)
# -----------------------------
def load_env_upwards(start_dir: Path) -> None:
    """
    Look for a .env file in start_dir and its parents; load KEY=VALUE lines into os.environ.
    This avoids requiring python-dotenv.
    """
    cur = start_dir.resolve()
    for _ in range(10):
        env_path = cur / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and v and k not in os.environ:
                    os.environ[k] = v
            return
        cur = cur.parent


# -----------------------------
# Helpers
# -----------------------------
def parse_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def normalize_interval(interval: Optional[str]) -> Optional[str]:
    """
    Pandas is deprecating 'T' in favor of 'min'. Accept both.
    Examples:
      '5T'  -> '5min'
      'T'   -> 'min'
      'H'   -> 'H'
      '5min' stays
    """
    if not interval:
        return None
    s = interval.strip()
    if s == "T":
        return "min"
    if s.endswith("T") and len(s) > 1 and s[:-1].isdigit():
        return f"{s[:-1]}min"
    return s


def choose_map_style(user_style: Optional[str], token: Optional[str]) -> str:
    if user_style:
        return user_style
    # "open-street-map" does not require a token
    return "dark" if token else "open-street-map"


def round_key(lat: float, lon: float, granularity: int) -> Tuple[str, str]:
    lat_key = f"{round(lat, granularity):.{granularity}f}"
    lon_key = f"{round(lon, granularity):.{granularity}f}"
    return lat_key, lon_key


def key_to_latlon(lat_key: str, lon_key: str) -> Tuple[float, float]:
    return float(lat_key), float(lon_key)

def rotate_latlon_about_pivot(lat: float, lon: float, pivot_lat: float, pivot_lon: float, angle_deg: float) -> Tuple[float, float]:
    """
    Rotate a lat/lon point around a pivot by angle_deg (CCW positive).
    Uses a local tangent-plane (meters) approximation; works well for small areas (intersections).
    """
    if angle_deg == 0.0:
        return lat, lon

    R = 6371000.0
    ang = math.radians(angle_deg)

    # Convert degrees to local meters around pivot
    lat0 = math.radians(pivot_lat)
    x = math.radians(lon - pivot_lon) * math.cos(lat0) * R
    y = math.radians(lat - pivot_lat) * R

    # Rotate in meters
    xr = x * math.cos(ang) - y * math.sin(ang)
    yr = x * math.sin(ang) + y * math.cos(ang)

    # Convert back to lat/lon
    lat_r = pivot_lat + math.degrees(yr / R)
    lon_r = pivot_lon + math.degrees(xr / (R * math.cos(lat0)))

    return lat_r, lon_r



# -----------------------------
# Reservoir sampling (bounded memory)
# -----------------------------
@dataclass
class Reservoir:
    k: int
    n_seen: int = 0
    items: List[Tuple[float, float]] = None

    def __post_init__(self):
        if self.items is None:
            self.items = []

    def add(self, lat: float, lon: float) -> None:
        self.n_seen += 1
        if len(self.items) < self.k:
            self.items.append((lat, lon))
            return
        j = random.randrange(self.n_seen)
        if j < self.k:
            self.items[j] = (lat, lon)


# -----------------------------
# Mongo access
# -----------------------------
def get_client(mongo_uri: str) -> "pymongo.MongoClient":
    if pymongo is None:
        raise RuntimeError("pymongo is not installed. Run: pip install pymongo")

    return pymongo.MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=30_000,
        connectTimeoutMS=30_000,
        socketTimeoutMS=120_000,
    )


def resolve_collection(db, name: str):
    """
    Some of your project uses 'combined_stats' vs 'combined-stats'. Try to be helpful.
    """
    existing = set(db.list_collection_names())
    if name in existing:
        return db[name]

    # try swapping '-' and '_'
    alt1 = name.replace("-", "_")
    alt2 = name.replace("_", "-")
    for alt in (alt1, alt2):
        if alt in existing:
            print(f"ℹ️ Collection '{name}' not found; using '{alt}' instead.")
            return db[alt]

    # fall back (will error later on find, but we give a clear message now)
    print(f"⚠️ Collection '{name}' not found. Collections available: {sorted(existing)[:30]} ...")
    return db[name]


def iter_docs_paged(
    client: "pymongo.MongoClient",
    db_name: str,
    collection_name: str,
    base_query: Dict[str, Any],
    projection: Dict[str, int],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    page_minutes: int,
    batch_size: int,
) -> Iterator[Dict[str, Any]]:
    """
    Avoid a single enormous cursor on Atlas tiers (and reduce timeout risk) by paging over time windows.
    If no start/end is provided, it does a single scan (not recommended for huge collections).
    """
    db = client[db_name]
    coll = resolve_collection(db, collection_name)

    if start is None or end is None:
        # Single-pass query
        cursor = coll.find(base_query, projection=projection).batch_size(batch_size)
        try:
            for doc in cursor:
                yield doc
        finally:
            cursor.close()
        return

    # Page by time window
    cur = start
    step = timedelta(minutes=page_minutes)

    while cur < end:
        nxt = min(end, cur + step)
        q = dict(base_query)
        q["timestamp"] = {"$gte": cur.to_pydatetime(), "$lt": nxt.to_pydatetime()}

        cursor = coll.find(q, projection=projection).batch_size(batch_size)
        try:
            for doc in cursor:
                yield doc
        finally:
            cursor.close()

        cur = nxt


# -----------------------------
# Main
# -----------------------------
def main():
    load_env_upwards(Path.cwd())

    ap = argparse.ArgumentParser(description="MongoDB car density heatmap (binned or raw-point).")

    # Mongo connection
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI", ""),
                    help="MongoDB URI. Can also set MONGO_URI env var.")
    ap.add_argument("--db", default=os.getenv("MONGO_DB", "camera-counts"), help="Database name.")
    ap.add_argument("--collection", default=os.getenv("MONGO_COLL", "combined-stats"),
                    help="Collection name (often combined-stats or combined_stats).")

    # Filters
    ap.add_argument("--location", default=None, help="Filter by location (e.g., patterson).")
    ap.add_argument("--only-type", dest="only_type", default="car",
                    help="Filter detected_type (case-insensitive). Use 'any' to disable. Default: car")
    ap.add_argument("--confident-only", action="store_true",
                    help="If set, only keep docs where is_confident == True.")
    ap.add_argument("--min-speed", type=float, default=None,
                    help="If set, only keep docs with speed_mps >= this value.")

    # Time range
    ap.add_argument("--start", type=str, default=None, help="Start datetime/date (UTC assumed).")
    ap.add_argument("--end", type=str, default=None, help="End datetime/date (exclusive).")

    # Heatmap time binning (animation)
    ap.add_argument("--interval", type=str, default=None,
                    help="If set (e.g., 5min, 15min, H), also produce an animated heatmap by time bins.")

    # Binning behavior
    ap.add_argument("--no-binning", action="store_true",
                    help="If set, do NOT round into a grid. Uses raw points (sampled) for a smoother heatmap.")
    ap.add_argument("--granularity", type=int, default=5,
                    help="Decimal places for rounding lat/lon if binning is enabled. Default 5.")

    # Performance
    ap.add_argument("--batch-size", type=int, default=50_000, help="Mongo cursor batch size. Default 50k.")
    ap.add_argument("--page-minutes", type=int, default=60,
                    help="Query in time windows of this many minutes (reduces giant cursor risk). Default 60.")
    ap.add_argument("--max-bins", type=int, default=500_000,
                    help="Safety cap on unique bins in binned mode. Default 500k.")
    ap.add_argument("--max-points", type=int, default=250_000,
                    help="Max points kept in memory in --no-binning static mode (reservoir sample).")
    ap.add_argument("--max-points-per-frame", type=int, default=30_000,
                    help="Max points per time frame in --no-binning animated mode (reservoir sample).")

    # Output + map
    ap.add_argument("--outdir", default="charts", help="Output directory (default: charts).")
    ap.add_argument("--map-style", default=None,
                    help="Plotly map style. Default open-street-map if no token; otherwise 'dark'.")
    ap.add_argument("--zoom", type=float, default=18.3, help="Map zoom level. Default 16.5.")
    ap.add_argument("--radius", type=int, default=4,
                    help="Density radius (px). Larger = smoother. Default 12.")
    ap.add_argument("--opacity", type=float, default=0.85, help="Heatmap opacity. Default 0.85.")

    # Camera-like view controls (Mapbox)
    ap.add_argument("--pitch", type=float, default=0.0, help="Map pitch (tilt) in degrees. Default 0.")
    ap.add_argument("--bearing", type=float, default=0.0, help="Map bearing (rotation) in degrees. Default 0.")

    # Center offsets (move the map center)
    ap.add_argument("--center-lat-offset", type=float, default=0.0,
                    help="Add this (degrees) to the computed map center latitude.")
    ap.add_argument("--center-lon-offset", type=float, default=0.0,
                    help="Add this (degrees) to the computed map center longitude.")

    # Point offsets (move plotted points WITHOUT moving the map)
    ap.add_argument("--points-lat-offset", type=float, default=0.0,
                    help="Add this (degrees) to every plotted latitude (shifts heatmap points).")
    ap.add_argument("--points-lon-offset", type=float, default=-0.0002,
                    help="Add this (degrees) to every plotted longitude (shifts heatmap points).")
    
    # args for rotating the points around the center
    ap.add_argument("--rotate-points-deg", type=float, default=0.0,
                    help="Rotate plotted points around the pivot by this many degrees (CCW+).")
    ap.add_argument("--pivot-lat", type=float, default=None,
                    help="Pivot latitude for point rotation. Default: computed map center.")
    ap.add_argument("--pivot-lon", type=float, default=None,
                    help="Pivot longitude for point rotation. Default: computed map center.")


    args = ap.parse_args()

    if not args.mongo_uri:
        raise ValueError(
            "Missing --mongo-uri (or MONGO_URI env var). Example:\n"
            '  --mongo-uri "mongodb+srv://USER:PASS@.../?"'
        )

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    interval = normalize_interval(args.interval)

    # Build base Mongo query
    query: Dict[str, Any] = {}
    if args.location:
        query["location"] = args.location

    if args.confident_only:
        query["is_confident"] = True

    if args.only_type.lower() != "any":
        query["detected_type"] = {"$regex": f"^{args.only_type}$", "$options": "i"}

    if args.min_speed is not None:
        query["speed_mps"] = {"$gte": float(args.min_speed)}

    projection = {
        "_id": 0,
        "timestamp": 1,
        "lat": 1,
        "lon": 1,
        "location": 1,
        "detected_type": 1,
        "is_confident": 1,
        "speed_mps": 1,
    }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Mapbox token optional
    token = os.getenv("MAPBOX_ACCESS_TOKEN") or os.getenv("MAPBOX_TOKEN")
    map_style = choose_map_style(args.map_style, token)
    if token:
        px.set_mapbox_access_token(token)

    client = get_client(args.mongo_uri)

    total_seen = 0
    total_used = 0

    # -------------------------
    # MODE B: no-binning (raw points, sampled)
    # -------------------------
    if args.no_binning:
        if interval:
            # Per-frame reservoirs (bounded)
            frames: Dict[pd.Timestamp, Reservoir] = {}
            frame_counts: Dict[pd.Timestamp, int] = {}

            for doc in iter_docs_paged(
                client=client,
                db_name=args.db,
                collection_name=args.collection,
                base_query=query,
                projection=projection,
                start=start,
                end=end,
                page_minutes=args.page_minutes,
                batch_size=args.batch_size,
            ):
                total_seen += 1

                lat = doc.get("lat", None)
                lon = doc.get("lon", None)
                ts = doc.get("timestamp", None)
                if lat is None or lon is None or ts is None:
                    continue

                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                except Exception:
                    continue

                ts_pd = pd.to_datetime(ts, utc=True, errors="coerce")
                if pd.isna(ts_pd):
                    continue

                # time bin
                tbin = ts_pd.floor(interval)

                if tbin not in frames:
                    frames[tbin] = Reservoir(k=int(args.max_points_per_frame))
                    frame_counts[tbin] = 0

                # apply POINT OFFSETS here (shift plotted points)
                frames[tbin].add(lat_f + float(args.points_lat_offset), lon_f + float(args.points_lon_offset))
                frame_counts[tbin] += 1
                total_used += 1

            client.close()

            if not frames:
                print("⚠️ No data found after filtering.")
                return

            # Build plotting DF from sampled points
            rows = []
            for tbin, res in sorted(frames.items(), key=lambda kv: kv[0]):
                # each sampled point represents 1 sample (weight=1)
                for (latp, lonp) in res.items:
                    rows.append({"time_bin": tbin, "lat_plot": latp, "lon_plot": lonp, "weight": 1})

            df = pd.DataFrame(rows)
            if df.empty:
                print("⚠️ No points retained (unexpected). Try increasing --max-points-per-frame.")
                return

            # Compute MAP CENTER from UN-shifted center of plotted points minus offsets,
            # or simply compute from plotted points and then undo point offset.
            # Easier: compute from plotted points, then apply center offsets.
            center_lat = float(df["lat_plot"].mean()) + float(args.center_lat_offset)
            center_lon = float(df["lon_plot"].mean()) + float(args.center_lon_offset)
            
            # --- rotate POINTS only (after offsets, before plotting) ---
            pivot_lat = float(args.pivot_lat) if args.pivot_lat is not None else float(df["lat_plot"].mean())
            pivot_lon = float(args.pivot_lon) if args.pivot_lon is not None else float(df["lon_plot"].mean())

            if float(args.rotate_points_deg) != 0.0:
                rotated = df.apply(
                    lambda r: rotate_latlon_about_pivot(
                        float(r["lat_plot"]), float(r["lon_plot"]),
                        pivot_lat, pivot_lon,
                        float(args.rotate_points_deg)
                    ),
                    axis=1,
                )
                df["lat_plot"] = [p[0] for p in rotated]
                df["lon_plot"] = [p[1] for p in rotated]


            df["time_frame"] = pd.to_datetime(df["time_bin"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

            # Static (collapsed) from sampled points
            fig_static = px.density_mapbox(
                df,
                lat="lat_plot",
                lon="lon_plot",
                z="weight",
                radius=int(args.radius),
                center={"lat": center_lat, "lon": center_lon},
                zoom=float(args.zoom),
                mapbox_style=map_style,
                title="Car density heatmap (no-binning; sampled raw points)",
                opacity=float(args.opacity),
            )
            fig_static.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

            out_static_html = outdir / "car_heatmap_static.html"
            out_static_json = outdir / "car_heatmap_static.json"
            fig_static.write_html(str(out_static_html), include_plotlyjs="cdn", full_html=True)

            # JSON = points used to plot (sampled)
            df_static_json = df[["lat_plot", "lon_plot", "weight"]].rename(
                columns={"lat_plot": "lat", "lon_plot": "lon"}
            )
            out_static_json.write_text(df_static_json.to_json(orient="records"), encoding="utf-8")

            print(f"✅ Docs seen: {total_seen:,} | used: {total_used:,}")
            print("✅ Wrote:")
            print(" ", out_static_html.resolve())
            print(" ", out_static_json.resolve())

            # Animated
            fig_ts = px.density_mapbox(
                df,
                lat="lat_plot",
                lon="lon_plot",
                z="weight",
                radius=int(args.radius),
                center={"lat": center_lat, "lon": center_lon},
                zoom=float(args.zoom),
                mapbox_style=map_style,
                title=f"Car density heatmap over time (bin={interval}, no-binning; sampled)",
                opacity=float(args.opacity),
                animation_frame="time_frame",
            )
            fig_ts.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

            out_ts_html = outdir / "car_heatmap_timeseries.html"
            out_ts_json = outdir / "car_heatmap_timeseries.json"
            fig_ts.write_html(str(out_ts_html), include_plotlyjs="cdn", full_html=True)

            # JSON = points per frame
            out_json = df[["time_frame", "lat_plot", "lon_plot", "weight"]].rename(
                columns={"time_frame": "time_bin", "lat_plot": "lat", "lon_plot": "lon"}
            )
            out_ts_json.write_text(out_json.to_json(orient="records"), encoding="utf-8")

            print(" ", out_ts_html.resolve())
            print(" ", out_ts_json.resolve())
            return

        else:
            # Static-only no-binning
            res = Reservoir(k=int(args.max_points))

            for doc in iter_docs_paged(
                client=client,
                db_name=args.db,
                collection_name=args.collection,
                base_query=query,
                projection=projection,
                start=start,
                end=end,
                page_minutes=args.page_minutes,
                batch_size=args.batch_size,
            ):
                total_seen += 1

                lat = doc.get("lat", None)
                lon = doc.get("lon", None)
                ts = doc.get("timestamp", None)
                if lat is None or lon is None or ts is None:
                    continue

                try:
                    lat_f = float(lat)
                    lon_f = float(lon)
                except Exception:
                    continue

                # apply POINT OFFSETS here (shift plotted points)
                res.add(lat_f + float(args.points_lat_offset), lon_f + float(args.points_lon_offset))
                total_used += 1

            client.close()

            if not res.items:
                print("⚠️ No data found after filtering.")
                return

            df = pd.DataFrame(res.items, columns=["lat_plot", "lon_plot"])
            df["weight"] = 1

            center_lat = float(df["lat_plot"].mean()) + float(args.center_lat_offset)
            center_lon = float(df["lon_plot"].mean()) + float(args.center_lon_offset)
            
            # --- rotate POINTS only (after offsets, before plotting) ---
            pivot_lat = float(args.pivot_lat) if args.pivot_lat is not None else float(df["lat_plot"].mean())
            pivot_lon = float(args.pivot_lon) if args.pivot_lon is not None else float(df["lon_plot"].mean())

            if float(args.rotate_points_deg) != 0.0:
                rotated = df.apply(
                    lambda r: rotate_latlon_about_pivot(
                        float(r["lat_plot"]), float(r["lon_plot"]),
                        pivot_lat, pivot_lon,
                        float(args.rotate_points_deg)
                    ),
                    axis=1,
                )
                df["lat_plot"] = [p[0] for p in rotated]
                df["lon_plot"] = [p[1] for p in rotated]


            fig_static = px.density_mapbox(
                df,
                lat="lat_plot",
                lon="lon_plot",
                z="weight",
                radius=int(args.radius),
                center={"lat": center_lat, "lon": center_lon},
                zoom=float(args.zoom),
                mapbox_style=map_style,
                title="Car density heatmap (no-binning; sampled raw points)",
                opacity=float(args.opacity),
            )
            fig_static.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

            out_static_html = outdir / "car_heatmap_static.html"
            out_static_json = outdir / "car_heatmap_static.json"
            fig_static.write_html(str(out_static_html), include_plotlyjs="cdn", full_html=True)

            df_json = df[["lat_plot", "lon_plot", "weight"]].rename(columns={"lat_plot": "lat", "lon_plot": "lon"})
            out_static_json.write_text(df_json.to_json(orient="records"), encoding="utf-8")

            print(f"✅ Docs seen: {total_seen:,} | used: {total_used:,} | points kept: {len(df):,}")
            print("✅ Wrote:")
            print(" ", out_static_html.resolve())
            print(" ", out_static_json.resolve())
            return

    # -------------------------
    # MODE A: binning (rounded grid)
    # -------------------------
    agg = {}  # key -> count
    # key is either (lat_key, lon_key) or (time_bin, lat_key, lon_key)

    for doc in iter_docs_paged(
        client=client,
        db_name=args.db,
        collection_name=args.collection,
        base_query=query,
        projection=projection,
        start=start,
        end=end,
        page_minutes=args.page_minutes,
        batch_size=args.batch_size,
    ):
        total_seen += 1

        lat = doc.get("lat", None)
        lon = doc.get("lon", None)
        ts = doc.get("timestamp", None)
        if lat is None or lon is None or ts is None:
            continue

        try:
            lat_f = float(lat)
            lon_f = float(lon)
        except Exception:
            continue

        ts_pd = pd.to_datetime(ts, utc=True, errors="coerce")
        if pd.isna(ts_pd):
            continue

        lat_key, lon_key = round_key(lat_f, lon_f, args.granularity)

        if interval:
            tbin = ts_pd.floor(interval)
            key = (tbin, lat_key, lon_key)
        else:
            key = (lat_key, lon_key)

        agg[key] = agg.get(key, 0) + 1
        total_used += 1

        if len(agg) > args.max_bins:
            print(f"⚠️ Reached max bins ({args.max_bins}). "
                  f"Reduce time range, lower granularity (e.g., 4), or disable binning (--no-binning).")
            break

    client.close()

    if not agg:
        print("⚠️ No data found after filtering.")
        return

    # Build df with lat_bin/lon_bin (unshifted) and also lat_plot/lon_plot (shifted)
    if interval:
        rows = []
        for (tbin, lat_key, lon_key), cnt in agg.items():
            lat_bin, lon_bin = key_to_latlon(lat_key, lon_key)
            rows.append({
                "timestamp": tbin,
                "time_bin": tbin,
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "count": int(cnt),
            })
        df = pd.DataFrame(rows).sort_values("time_bin")
    else:
        rows = []
        for (lat_key, lon_key), cnt in agg.items():
            lat_bin, lon_bin = key_to_latlon(lat_key, lon_key)
            rows.append({
                "lat_bin": lat_bin,
                "lon_bin": lon_bin,
                "count": int(cnt),
            })
        df = pd.DataFrame(rows)

    # Shift points WITHOUT shifting center
    df["lat_plot"] = pd.to_numeric(df["lat_bin"], errors="coerce") + float(args.points_lat_offset)
    df["lon_plot"] = pd.to_numeric(df["lon_bin"], errors="coerce") + float(args.points_lon_offset)
    df = df.dropna(subset=["lat_plot", "lon_plot"])

    # Map center uses UN-shifted bins (then apply center offsets)
    center_lat = float(pd.to_numeric(df["lat_bin"], errors="coerce").dropna().mean()) + float(args.center_lat_offset)
    center_lon = float(pd.to_numeric(df["lon_bin"], errors="coerce").dropna().mean()) + float(args.center_lon_offset)
    
    # --- rotate POINTS only (after offsets, before plotting) ---
    pivot_lat = float(args.pivot_lat) if args.pivot_lat is not None else float(df["lat_plot"].mean())
    pivot_lon = float(args.pivot_lon) if args.pivot_lon is not None else float(df["lon_plot"].mean())

    if float(args.rotate_points_deg) != 0.0:
        rotated = df.apply(
            lambda r: rotate_latlon_about_pivot(
                float(r["lat_plot"]), float(r["lon_plot"]),
                pivot_lat, pivot_lon,
                float(args.rotate_points_deg)
            ),
            axis=1,
        )
        df["lat_plot"] = [p[0] for p in rotated]
        df["lon_plot"] = [p[1] for p in rotated]


    fig_static = px.density_mapbox(
        df if not interval else df.drop(columns=["time_bin", "timestamp"], errors="ignore"),
        lat="lat_plot",
        lon="lon_plot",
        z="count",
        radius=int(args.radius),
        center={"lat": center_lat, "lon": center_lon},
        zoom=float(args.zoom),
        mapbox_style=map_style,
        title=f"Car density heatmap (BINNED {args.granularity}dp grid)",
        opacity=float(args.opacity),
    )
    fig_static.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

    out_static_html = outdir / "car_heatmap_static.html"
    out_static_json = outdir / "car_heatmap_static.json"
    fig_static.write_html(str(out_static_html), include_plotlyjs="cdn", full_html=True)

    # JSON output is the plotted bins (shifted) so it matches what you see
    df_json = df[["lat_plot", "lon_plot", "count"]].rename(columns={"lat_plot": "lat", "lon_plot": "lon", "count": "weight"})
    out_static_json.write_text(df_json.to_json(orient="records"), encoding="utf-8")

    print(f"✅ Docs seen: {total_seen:,} | used: {total_used:,} | unique bins: {len(df):,}")
    print("✅ Wrote:")
    print(" ", out_static_html.resolve())
    print(" ", out_static_json.resolve())

    if interval:
        df_anim = df.copy()
        df_anim["time_frame"] = pd.to_datetime(df_anim["time_bin"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

        fig_ts = px.density_mapbox(
            df_anim,
            lat="lat_plot",
            lon="lon_plot",
            z="count",
            radius=int(args.radius),
            center={"lat": center_lat, "lon": center_lon},
            zoom=float(args.zoom),
            mapbox_style=map_style,
            title=f"Car density heatmap over time (bin={interval}, binned {args.granularity}dp)",
            opacity=float(args.opacity),
            animation_frame="time_frame",
        )
        fig_ts.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

        out_ts_html = outdir / "car_heatmap_timeseries.html"
        out_ts_json = outdir / "car_heatmap_timeseries.json"
        fig_ts.write_html(str(out_ts_html), include_plotlyjs="cdn", full_html=True)

        out_json = df_anim[["time_frame", "lat_plot", "lon_plot", "count"]].rename(
            columns={"time_frame": "time_bin", "lat_plot": "lat", "lon_plot": "lon", "count": "weight"}
        )
        out_ts_json.write_text(out_json.to_json(orient="records"), encoding="utf-8")

        print(" ", out_ts_html.resolve())
        print(" ", out_ts_json.resolve())


if __name__ == "__main__":
    main()
