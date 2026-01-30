#!/usr/bin/env python3
"""
Car heatmap (spatial density) from a big CSV via chunking.

Inputs expected (case-insensitive headers supported):
- timestamp
- detected_type (optional but recommended)
- lat, lon

Outputs (default ./charts): 
- car_heatmap_static.html
- car_heatmap_static.json
Optionally (if --interval is provided):
- car_heatmap_timeseries.html
- car_heatmap_timeseries.json

Examples:
  python car_heatmap.py --csv ..\\bosch-metadata-reader\\combined_vehicle_stats_with_derivatives.csv --start 2025-10-07 --end 2025-10-08
  python car_heatmap.py --csv your.csv --interval 15T --bin-meters 8
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from dotenv import load_dotenv
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import plotly.express as px


# ---------------------------
# Helpers: robust column finding
# ---------------------------

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"   # CPE350/.env
load_dotenv(ENV_PATH)

MAPBOX_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")  # Mapbox token from .env
if not MAPBOX_TOKEN:
    raise RuntimeError("MAPBOX_ACCESS_TOKEN is missing in .env. Tried loading {ENV_PATH}")

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

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(s).lstrip("\ufeff").strip().lower())

def sniff_delimiter(path: Path) -> str:
    sample = path.read_text(encoding="utf-8-sig", errors="replace")[:4096]
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"]).delimiter
    except Exception:
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get)

def pick_column(cols: list[str], candidates: list[str]) -> Optional[str]:
    m = {norm(c): c for c in cols}
    for cand in candidates:
        k = norm(cand)
        if k in m:
            return m[k]
    return None

def parse_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


# ---------------------------
# Bin sizing: meters -> degrees
# ---------------------------

def meters_to_deg_lat(m: float) -> float:
    # ~111,320 meters per degree latitude
    return m / 111_320.0

def meters_to_deg_lon(m: float, at_lat_deg: float) -> float:
    # lon degrees shrink by cos(latitude)
    denom = 111_320.0 * max(1e-6, math.cos(math.radians(at_lat_deg)))
    return m / denom

def grid_bin(lat: float, lon: float, dlat: float, dlon: float) -> Tuple[float, float]:
    # Use floor-binning (stable) and return bin center
    lat_bin = (math.floor(lat / dlat) + 0.5) * dlat
    lon_bin = (math.floor(lon / dlon) + 0.5) * dlon
    return lat_bin, lon_bin


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Create a car-density heatmap over an intersection.")
    ap.add_argument("--csv", required=True, help="Path to CSV.")
    ap.add_argument("--outdir", default="charts", help="Output directory (default: charts)")
    ap.add_argument("--only-type", dest="only_type", default="car",
                    help="Filter detected_type (case-insensitive). Use 'any' to disable. Default: car")
    ap.add_argument("--start", type=str, default=None, help="Start datetime/date (UTC assumed if no tz).")
    ap.add_argument("--end", type=str, default=None, help="End datetime/date (exclusive).")
    ap.add_argument("--interval", type=str, default=None,
                    help="If set (e.g., H, 15T), also produce an animated heatmap by time bins.")
    ap.add_argument("--bin-meters", type=float, default=8.0,
                    help="Spatial bin size in meters (smaller = finer heatmap). Default 8m")
    ap.add_argument("--chunk-size", type=int, default=500_000, help="CSV chunk size. Default 500k")
    ap.add_argument("--max-bins", type=int, default=300_000,
                    help="Safety cap on number of unique spatial bins (or time+spatial bins). Default 300k")
    ap.add_argument("--map-style", default=None,
                    help="Plotly map style. Default open-street-map (no token). "
                         "If MAPBOX_ACCESS_TOKEN exists and this is None, uses 'dark'.")
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sep = sniff_delimiter(path)

    # Detect columns from header
    header = pd.read_csv(path, nrows=0, sep=sep, encoding="utf-8-sig")
    cols = list(header.columns)

    TS  = pick_column(cols, ["timestamp", "time", "utc_time", "datetime", "ts"])
    LAT = pick_column(cols, ["lat", "latitude"])
    LON = pick_column(cols, ["lon", "lng", "longitude"])
    TYP = pick_column(cols, ["detected_type", "type", "class", "label"])

    missing = [name for name, col in [("timestamp", TS), ("lat", LAT), ("lon", LON)] if col is None]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\nColumns seen: {cols}")

    usecols = [TS, LAT, LON] + ([TYP] if TYP else [])

    # Time filters
    start = parse_dt(args.start)
    end = parse_dt(args.end)

    # Map style + token (optional)
    token = os.getenv("MAPBOX_ACCESS_TOKEN") or os.getenv("MAPBOX_TOKEN")
    if args.map_style:
        map_style = args.map_style
    else:
        # No token required:
        # - "open-street-map" works without tokens
        # If token exists, you can use "dark", "satellite", etc.
        map_style = "dark" if token else "open-street-map"

    if token:
        px.set_mapbox_access_token(token)

    # We need a latitude to convert meters->deg lon; use a small sample from the CSV
    sample_df = pd.read_csv(path, sep=sep, encoding="utf-8-sig", usecols=[LAT], nrows=50_000)
    sample_df = sample_df.dropna()
    if sample_df.empty:
        raise ValueError("No latitude values found to estimate bin sizing.")
    lat_ref = float(sample_df[LAT].astype("float64").mean())

    dlat = meters_to_deg_lat(args.bin_meters)
    dlon = meters_to_deg_lon(args.bin_meters, lat_ref)

    # Aggregation dict:
    # - static: key=(lat_bin, lon_bin) -> count
    # - animated: key=(time_bin, lat_bin, lon_bin) -> count
    agg = defaultdict(int)

    # Stream CSV
    for chunk in pd.read_csv(
        path,
        sep=sep,
        encoding="utf-8-sig",
        usecols=usecols,
        chunksize=args.chunk_size,
        engine="c",
        low_memory=True,
    ):
        # optional type filter
        if args.only_type.lower() != "any" and TYP and (TYP in chunk.columns):
            chunk = chunk[chunk[TYP].astype("string").str.lower() == args.only_type.lower()]
            if chunk.empty:
                continue

        # drop bad rows
        chunk = chunk.dropna(subset=[TS, LAT, LON])
        if chunk.empty:
            continue

        # parse time
        chunk[TS] = pd.to_datetime(chunk[TS], utc=True, errors="coerce")
        chunk = chunk[chunk[TS].notna()]
        if chunk.empty:
            continue

        if start is not None:
            chunk = chunk[chunk[TS] >= start]
        if end is not None:
            chunk = chunk[chunk[TS] < end]
        if chunk.empty:
            continue

        # numeric lat/lon
        chunk[LAT] = pd.to_numeric(chunk[LAT], errors="coerce")
        chunk[LON] = pd.to_numeric(chunk[LON], errors="coerce")
        chunk = chunk[chunk[LAT].notna() & chunk[LON].notna()]
        if chunk.empty:
            continue

        if args.interval:
            chunk["time_bin"] = chunk[TS].dt.floor(args.interval)

        # aggregate row-by-row (fast enough after chunking; bins keep it bounded)
        for row in chunk.itertuples(index=False):
            # row fields in order of usecols (+time_bin)
            ts_val = getattr(row, TS)
            lat = float(getattr(row, LAT))
            lon = float(getattr(row, LON))
            lat_b, lon_b = grid_bin(lat, lon, dlat, dlon)

            if args.interval:
                tbin = getattr(row, "time_bin")
                key = (tbin, lat_b, lon_b)
            else:
                key = (lat_b, lon_b)

            agg[key] += 1

        # safety: prevent runaway memory if bins explode
        if len(agg) > args.max_bins:
            print(f"⚠️ Reached max bins ({args.max_bins}). "
                  f"Increase --bin-meters or decrease time range / use larger --interval.")
            break

    # Build final DataFrame
    if not agg:
        print("⚠️ No data after filters. Try widening time range or using --only-type any.")
        return

    if args.interval:
        rows = [{"time_bin": k[0], "lat_bin": k[1], "lon_bin": k[2], "count": v} for k, v in agg.items()]
    else:
        rows = [{"lat_bin": k[0], "lon_bin": k[1], "count": v} for k, v in agg.items()]

    df = pd.DataFrame(rows)

    # Determine map center
    center_lat = float(df["lat_bin"].mean())
    center_lon = float(df["lon_bin"].mean())

    # --------------------
    # Static heatmap
    # --------------------
    # We render bins as points with a density kernel via Plotly's density_mapbox
    fig_static = px.density_mapbox(
        df,
        lat="lat_bin",
        lon="lon_bin",
        z="count",
        radius=15,  # pixels; adjust if you want more/less smoothing
        center={"lat": center_lat, "lon": center_lon},
        zoom=17,
        mapbox_style=map_style,
        title="Car density heatmap (binned counts)",
        opacity=0.85,
    )

    out_static_html = outdir / "car_heatmap_static.html"
    out_static_json = outdir / "car_heatmap_static.json"
    fig_static.write_html(str(out_static_html), include_plotlyjs="cdn", full_html=True)
    out_static_json.write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")

    print("✅ Wrote:")
    print(" ", out_static_html.resolve())
    print(" ", out_static_json.resolve())

    # --------------------
    # Animated heatmap (optional)
    # --------------------
    if args.interval:
        df["time_bin"] = pd.to_datetime(df["time_bin"], utc=True)
        df = df.sort_values("time_bin")

        fig_ts = px.density_mapbox(
            df,
            lat="lat_bin",
            lon="lon_bin",
            z="count",
            radius=15,
            center={"lat": center_lat, "lon": center_lon},
            zoom=17,
            mapbox_style=map_style,
            title=f"Car density heatmap over time (bin={args.interval})",
            opacity=0.85,
            animation_frame=df["time_bin"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        )

        out_ts_html = outdir / "car_heatmap_timeseries.html"
        out_ts_json = outdir / "car_heatmap_timeseries.json"
        fig_ts.write_html(str(out_ts_html), include_plotlyjs="cdn", full_html=True)
        out_ts_json.write_text(df.to_json(orient="records", date_format="iso"), encoding="utf-8")

        print(" ", out_ts_html.resolve())
        print(" ", out_ts_json.resolve())


if __name__ == "__main__":
    main()
