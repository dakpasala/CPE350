#!/usr/bin/env python3
"""
car_heatmapV3.py (FastAPI → spatial density heatmap)

Fetches combined_stats from:
  GET /stats/combined?time_range=...&limit=...&location=...

Then renders:
- Static heatmap
- Optional animated heatmap (--interval)

Supports:
- --no-binning (raw points; reservoir sample; smoother)
- OR binning by rounding (--granularity)

Mapbox token optional (MAPBOX_ACCESS_TOKEN).

Dependencies:
  pip install requests pandas plotly
"""

from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import plotly.express as px
import requests
import math


# -----------------------------
# .env loading (walk up parents)
# -----------------------------
def load_env_upwards(start_dir: Path) -> None:
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


def parse_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def normalize_interval(interval: Optional[str]) -> Optional[str]:
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
    return "dark" if token else "open-street-map"


def round_key(lat: float, lon: float, granularity: int) -> Tuple[float, float]:
    lat_r = round(lat, granularity)
    lon_r = round(lon, granularity)
    return float(lat_r), float(lon_r)


def rotate_latlon_about_pivot(lat: float, lon: float, pivot_lat: float, pivot_lon: float, angle_deg: float) -> Tuple[float, float]:
    if angle_deg == 0.0:
        return lat, lon

    R = 6371000.0
    ang = math.radians(angle_deg)

    lat0 = math.radians(pivot_lat)
    x = math.radians(lon - pivot_lon) * math.cos(lat0) * R
    y = math.radians(lat - pivot_lat) * R

    xr = x * math.cos(ang) - y * math.sin(ang)
    yr = x * math.sin(ang) + y * math.cos(ang)

    lat_r = pivot_lat + math.degrees(yr / R)
    lon_r = pivot_lon + math.degrees(xr / (R * math.cos(lat0)))
    return lat_r, lon_r


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


def fetch_combined_stats(
    api_base: str,
    time_range: str,
    limit: int,
    location: Optional[str],
    timeout_s: int,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/stats/combined"
    params: Dict[str, Any] = {"time_range": time_range, "limit": int(limit)}
    if location:
        params["location"] = location

    resp = requests.get(url, params=params, timeout=timeout_s)
    resp.raise_for_status()
    return resp.json()


def main():
    load_env_upwards(Path.cwd())

    ap = argparse.ArgumentParser(description="Car heatmap from FastAPI endpoint (binned or raw-point).")

    ap.add_argument("--api-base", default="http://127.0.0.1:8000",
                    help="FastAPI base URL (default: http://127.0.0.1:8000)")
    ap.add_argument("--time-range", default="hour",
                    help='One of: "hour", "6hours", "12hours", "day", "week", "month" (default: hour)')
    ap.add_argument("--limit", type=int, default=250_000,
                    help="Max rows fetched from API (default: 250000)")
    ap.add_argument("--location", default=None, help="Filter by location (e.g., patterson)")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default: 60)")

    # Filters
    ap.add_argument("--only-type", dest="only_type", default="car",
                    help="Filter detected_type (case-insensitive). Use 'any' to disable. Default: car")
    ap.add_argument("--confident-only", action="store_true",
                    help="If set, only include is_confident==True (if column exists).")
    ap.add_argument("--min-speed", type=float, default=None,
                    help="If set, keep only rows with speed_mps >= this value")

    # Optional extra time slicing AFTER fetch
    ap.add_argument("--start", type=str, default=None, help="Optional start datetime/date (UTC) applied after fetch.")
    ap.add_argument("--end", type=str, default=None, help="Optional end datetime/date (exclusive, UTC) applied after fetch.")

    # Heatmap time binning (animation)
    ap.add_argument("--interval", type=str, default=None,
                    help="If set (e.g., 5min, 15min, H), produce animated heatmap by time bins.")

    # Binning behavior
    ap.add_argument("--no-binning", action="store_true",
                    help="If set, do NOT round into a grid. Uses raw points (sampled) for smoother look.")
    ap.add_argument("--granularity", type=int, default=5,
                    help="Decimal places for rounding lat/lon if binning is enabled. Default 5.")

    # Sampling limits (no-binning)
    ap.add_argument("--max-points", type=int, default=250_000,
                    help="Max points kept in memory in --no-binning static mode.")
    ap.add_argument("--max-points-per-frame", type=int, default=30_000,
                    help="Max points per frame in --no-binning animated mode.")

    # Output + map
    ap.add_argument("--outdir", default="charts", help="Output directory (default: charts)")
    ap.add_argument("--map-style", default=None,
                    help="Plotly map style. Default open-street-map if no token; otherwise 'dark'.")
    ap.add_argument("--zoom", type=float, default=18.3, help="Map zoom level.")
    ap.add_argument("--radius", type=int, default=4, help="Density radius (px). Larger = smoother.")
    ap.add_argument("--opacity", type=float, default=0.85, help="Heatmap opacity.")

    # Camera-like view controls (Mapbox)
    ap.add_argument("--pitch", type=float, default=0.0, help="Map pitch (tilt) in degrees.")
    ap.add_argument("--bearing", type=float, default=0.0, help="Map bearing (rotation) in degrees.")

    # Center offsets (move map center)
    ap.add_argument("--center-lat-offset", type=float, default=0.0,
                    help="Add this (degrees) to computed map center latitude.")
    ap.add_argument("--center-lon-offset", type=float, default=0.0,
                    help="Add this (degrees) to computed map center longitude.")

    # Point offsets (move plotted points w/o moving center)
    ap.add_argument("--points-lat-offset", type=float, default=0.0,
                    help="Add this (degrees) to every plotted latitude.")
    ap.add_argument("--points-lon-offset", type=float, default=0.0,
                    help="Add this (degrees) to every plotted longitude.")

    # Optional: rotate points (NOT the map)
    ap.add_argument("--rotate-points-deg", type=float, default=0.0,
                    help="Rotate plotted points around pivot by this many degrees (CCW+).")
    ap.add_argument("--pivot-lat", type=float, default=None,
                    help="Pivot latitude for point rotation. Default: computed map center.")
    ap.add_argument("--pivot-lon", type=float, default=None,
                    help="Pivot longitude for point rotation. Default: computed map center.")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    start = parse_dt(args.start)
    end = parse_dt(args.end)
    interval = normalize_interval(args.interval)

    payload = fetch_combined_stats(
        api_base=args.api_base,
        time_range=args.time_range,
        limit=args.limit,
        location=args.location,
        timeout_s=args.timeout,
    )

    n = int(payload.get("count", 0))
    if n == 0 or not payload.get("data"):
        print("⚠️ API returned no data.")
        return

    if n >= int(args.limit):
        print(f"⚠️ API returned count={n:,} and limit={args.limit:,}. "
              f"Data may be truncated; consider increasing --limit.")

    df = pd.DataFrame(payload["data"])
    if df.empty:
        print("⚠️ Empty dataframe after parsing API data.")
        return

    # Require lat/lon/timestamp
    required = {"timestamp", "lat", "lon"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Have: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df[df["timestamp"].notna() & df["lat"].notna() & df["lon"].notna()]
    if df.empty:
        print("⚠️ No valid rows after parsing lat/lon/timestamp.")
        return

    # Optional filters
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] < end]
    if df.empty:
        print("⚠️ No data after --start/--end filtering.")
        return

    if args.only_type.lower() != "any" and "detected_type" in df.columns:
        df["detected_type"] = df["detected_type"].astype("string")
        df = df[df["detected_type"].str.lower() == args.only_type.lower()]
        if df.empty:
            print("⚠️ No data after --only-type filtering.")
            return

    if args.confident_only and "is_confident" in df.columns:
        df = df[df["is_confident"] == True]  # noqa: E712
        if df.empty:
            print("⚠️ No data after --confident-only filtering.")
            return

    if args.min_speed is not None and "speed_mps" in df.columns:
        df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce").fillna(0.0)
        df = df[df["speed_mps"] >= float(args.min_speed)]
        if df.empty:
            print("⚠️ No data after --min-speed filtering.")
            return

    # Map setup
    token = os.getenv("MAPBOX_ACCESS_TOKEN") or os.getenv("MAPBOX_TOKEN")
    map_style = choose_map_style(args.map_style, token)
    if token:
        px.set_mapbox_access_token(token)

    # Compute map center from raw points BEFORE point shift/rotation
    center_lat_raw = float(df["lat"].mean()) + float(args.center_lat_offset)
    center_lon_raw = float(df["lon"].mean()) + float(args.center_lon_offset)

    pivot_lat = float(args.pivot_lat) if args.pivot_lat is not None else center_lat_raw
    pivot_lon = float(args.pivot_lon) if args.pivot_lon is not None else center_lon_raw

    # Helper to apply transforms to points
    def transform_points(lat_s: pd.Series, lon_s: pd.Series) -> Tuple[pd.Series, pd.Series]:
        latp = lat_s.astype("float64") + float(args.points_lat_offset)
        lonp = lon_s.astype("float64") + float(args.points_lon_offset)

        if float(args.rotate_points_deg) != 0.0:
            out_lat = []
            out_lon = []
            for la, lo in zip(latp.to_list(), lonp.to_list()):
                la2, lo2 = rotate_latlon_about_pivot(la, lo, pivot_lat, pivot_lon, float(args.rotate_points_deg))
                out_lat.append(la2)
                out_lon.append(lo2)
            return pd.Series(out_lat), pd.Series(out_lon)

        return latp, lonp

    # -------------------------
    # MODE: no-binning (sampled raw points)
    # -------------------------
    if args.no_binning:
        if interval:
            # per-frame reservoir sampling
            frames: Dict[pd.Timestamp, Reservoir] = {}

            for r in df.itertuples(index=False):
                ts = getattr(r, "timestamp")
                lat = float(getattr(r, "lat"))
                lon = float(getattr(r, "lon"))
                tbin = ts.floor(interval)

                if tbin not in frames:
                    frames[tbin] = Reservoir(k=int(args.max_points_per_frame))

                # point transform
                latp, lonp = transform_points(pd.Series([lat]), pd.Series([lon]))
                frames[tbin].add(float(latp.iloc[0]), float(lonp.iloc[0]))

            rows = []
            for tbin, res in sorted(frames.items(), key=lambda kv: kv[0]):
                for (latp, lonp) in res.items:
                    rows.append({"time_bin": tbin, "lat_plot": latp, "lon_plot": lonp, "weight": 1})

            dfp = pd.DataFrame(rows)
            if dfp.empty:
                print("⚠️ No points retained; increase --max-points-per-frame.")
                return

            dfp["time_frame"] = pd.to_datetime(dfp["time_bin"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

            fig_static = px.density_mapbox(
                dfp,
                lat="lat_plot",
                lon="lon_plot",
                z="weight",
                radius=int(args.radius),
                center={"lat": center_lat_raw, "lon": center_lon_raw},
                zoom=float(args.zoom),
                mapbox_style=map_style,
                title="Car density heatmap (API, no-binning; sampled raw points)",
                opacity=float(args.opacity),
            )
            fig_static.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

            out_static_html = outdir / "car_heatmap_static.html"
            out_static_json = outdir / "car_heatmap_static.json"
            fig_static.write_html(str(out_static_html), include_plotlyjs="cdn", full_html=True)

            out_static_json.write_text(
                dfp[["lat_plot", "lon_plot", "weight"]].rename(columns={"lat_plot": "lat", "lon_plot": "lon"}).to_json(orient="records"),
                encoding="utf-8",
            )

            print("✅ Wrote:")
            print(" ", out_static_html.resolve())
            print(" ", out_static_json.resolve())

            fig_ts = px.density_mapbox(
                dfp,
                lat="lat_plot",
                lon="lon_plot",
                z="weight",
                radius=int(args.radius),
                center={"lat": center_lat_raw, "lon": center_lon_raw},
                zoom=float(args.zoom),
                mapbox_style=map_style,
                title=f"Car density heatmap over time (API, bin={interval}, no-binning; sampled)",
                opacity=float(args.opacity),
                animation_frame="time_frame",
            )
            fig_ts.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

            out_ts_html = outdir / "car_heatmap_timeseries.html"
            out_ts_json = outdir / "car_heatmap_timeseries.json"
            fig_ts.write_html(str(out_ts_html), include_plotlyjs="cdn", full_html=True)

            out_ts_json.write_text(
                dfp[["time_frame", "lat_plot", "lon_plot", "weight"]]
                .rename(columns={"time_frame": "time_bin", "lat_plot": "lat", "lon_plot": "lon"})
                .to_json(orient="records"),
                encoding="utf-8",
            )

            print(" ", out_ts_html.resolve())
            print(" ", out_ts_json.resolve())
            return

        # static-only no-binning
        res = Reservoir(k=int(args.max_points))
        latp_all, lonp_all = transform_points(df["lat"], df["lon"])
        for la, lo in zip(latp_all.to_list(), lonp_all.to_list()):
            res.add(float(la), float(lo))

        dfp = pd.DataFrame(res.items, columns=["lat_plot", "lon_plot"])
        dfp["weight"] = 1

        fig_static = px.density_mapbox(
            dfp,
            lat="lat_plot",
            lon="lon_plot",
            z="weight",
            radius=int(args.radius),
            center={"lat": center_lat_raw, "lon": center_lon_raw},
            zoom=float(args.zoom),
            mapbox_style=map_style,
            title="Car density heatmap (API, no-binning; sampled raw points)",
            opacity=float(args.opacity),
        )
        fig_static.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

        out_static_html = outdir / "car_heatmap_static.html"
        out_static_json = outdir / "car_heatmap_static.json"
        fig_static.write_html(str(out_static_html), include_plotlyjs="cdn", full_html=True)

        out_static_json.write_text(
            dfp[["lat_plot", "lon_plot", "weight"]].rename(columns={"lat_plot": "lat", "lon_plot": "lon"}).to_json(orient="records"),
            encoding="utf-8",
        )

        print("✅ Wrote:")
        print(" ", out_static_html.resolve())
        print(" ", out_static_json.resolve())
        return

    # -------------------------
    # MODE: binning (rounded grid)
    # -------------------------
    # Build bins (optionally per time bin)
    if interval:
        df["time_bin"] = df["timestamp"].dt.floor(interval)

    lat_bin, lon_bin = [], []
    for la, lo in zip(df["lat"].to_list(), df["lon"].to_list()):
        la2, lo2 = round_key(float(la), float(lo), int(args.granularity))
        lat_bin.append(la2)
        lon_bin.append(lo2)

    df["lat_bin"] = lat_bin
    df["lon_bin"] = lon_bin

    # Count bins
    if interval:
        g = df.groupby(["time_bin", "lat_bin", "lon_bin"], as_index=False).size().rename(columns={"size": "count"})
        g["time_frame"] = pd.to_datetime(g["time_bin"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    else:
        g = df.groupby(["lat_bin", "lon_bin"], as_index=False).size().rename(columns={"size": "count"})

    # Transform plotted points without moving center
    g["lat_plot"], g["lon_plot"] = transform_points(g["lat_bin"], g["lon_bin"])

    fig_static = px.density_mapbox(
        g if not interval else g.drop(columns=["time_bin", "time_frame"], errors="ignore"),
        lat="lat_plot",
        lon="lon_plot",
        z="count",
        radius=int(args.radius),
        center={"lat": center_lat_raw, "lon": center_lon_raw},
        zoom=float(args.zoom),
        mapbox_style=map_style,
        title=f"Car density heatmap (API, BINNED {args.granularity}dp)",
        opacity=float(args.opacity),
    )
    fig_static.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

    out_static_html = outdir / "car_heatmap_static.html"
    out_static_json = outdir / "car_heatmap_static.json"
    fig_static.write_html(str(out_static_html), include_plotlyjs="cdn", full_html=True)

    g_json = g[["lat_plot", "lon_plot", "count"]].rename(columns={"lat_plot": "lat", "lon_plot": "lon", "count": "weight"})
    out_static_json.write_text(g_json.to_json(orient="records"), encoding="utf-8")

    print("✅ Wrote:")
    print(" ", out_static_html.resolve())
    print(" ", out_static_json.resolve())

    if interval:
        fig_ts = px.density_mapbox(
            g,
            lat="lat_plot",
            lon="lon_plot",
            z="count",
            radius=int(args.radius),
            center={"lat": center_lat_raw, "lon": center_lon_raw},
            zoom=float(args.zoom),
            mapbox_style=map_style,
            title=f"Car density heatmap over time (API, bin={interval}, binned {args.granularity}dp)",
            opacity=float(args.opacity),
            animation_frame="time_frame",
        )
        fig_ts.update_layout(mapbox=dict(pitch=float(args.pitch), bearing=float(args.bearing)))

        out_ts_html = outdir / "car_heatmap_timeseries.html"
        out_ts_json = outdir / "car_heatmap_timeseries.json"
        fig_ts.write_html(str(out_ts_html), include_plotlyjs="cdn", full_html=True)

        out_ts_json.write_text(
            g[["time_frame", "lat_plot", "lon_plot", "count"]]
            .rename(columns={"time_frame": "time_bin", "lat_plot": "lat", "lon_plot": "lon", "count": "weight"})
            .to_json(orient="records"),
            encoding="utf-8",
        )

        print(" ", out_ts_html.resolve())
        print(" ", out_ts_json.resolve())


if __name__ == "__main__":
    main()
