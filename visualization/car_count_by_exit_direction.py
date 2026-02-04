#!/usr/bin/env python3
"""
Count cars by the direction they EXIT the frame (using the last K heading samples).

Outputs (default: ./charts):
- car_counts_by_exit_total.html / .json
- car_counts_by_exit_timeseries.html / .json

Run:
  python car_count_by_exit_direction.py --csv your.csv
  python car_count_by_exit_direction.py --csv your.csv --interval H --min-speed 0.2
  python car_count_by_exit_direction.py --csv your.csv --only-type any
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px

# ---------------------------
# Robust header helpers
# ---------------------------

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

# ---------------------------
# Direction math
# ---------------------------

def heading_to_cardinal4(deg: float) -> str:
    """0°=N, 90°=E, 180°=S, 270°=W."""
    d = deg % 360.0
    if d >= 315 or d < 45:
        return "N"
    if d < 135:
        return "E"
    if d < 225:
        return "S"
    return "W"

def circular_mean_deg(degs: list[float]) -> float:
    """Circular mean in degrees for angles on [0,360)."""
    if not degs:
        return float("nan")
    rad = [math.radians(d % 360.0) for d in degs]
    s = sum(math.sin(r) for r in rad)
    c = sum(math.cos(r) for r in rad)
    if s == 0 and c == 0:
        return float("nan")
    return (math.degrees(math.atan2(s, c)) + 360.0) % 360.0

# ---------------------------
# Track tail state
# ---------------------------

@dataclass
class TrackTail:
    last_ts: pd.Timestamp
    headings: deque  # deque[float], size <= K

# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Count cars by EXIT direction using last K heading samples.")
    ap.add_argument("--csv", required=True, help="Path to CSV.")
    ap.add_argument("--outdir", default="charts", help="Output directory (default: charts)")
    ap.add_argument("--only-type", dest="only_type", default="car",
                    help="Filter detected_type (case-insensitive). Use 'any' to disable. Default: car")
    ap.add_argument("--min-speed", type=float, default=0.2,
                    help="Ignore rows with speed_mps below this (m/s). Default 0.2")
    ap.add_argument("--heading-offset", type=float, default=0.0,
                    help="Rotate heading by this many degrees if camera axes are rotated. Default 0")
    ap.add_argument("--tail-max-samples", type=int, default=5,
                    help="How many last heading samples per object to keep. Default 5")
    ap.add_argument("--interval", default="H",
                    help="Timeseries bin size (H, 15T, T, D, etc). Default H")
    ap.add_argument("--chunk-size", type=int, default=500_000, help="CSV chunk size. Default 500k")
    args = ap.parse_args()

    path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sep = sniff_delimiter(path)

    # Header
    head = pd.read_csv(path, nrows=0, sep=sep, encoding="utf-8-sig")
    cols = list(head.columns)

    OBJ = pick_column(cols, ["object_id", "track_id", "id", "uuid"])
    TS  = pick_column(cols, ["timestamp", "time", "utc_time", "datetime", "ts"])
    HDG = pick_column(cols, ["heading_deg", "heading", "bearing_deg", "bearing"])
    SPD = pick_column(cols, ["speed_mps", "speed", "velocity_mps", "velocity"])
    TYP = pick_column(cols, ["detected_type", "type", "class", "label"])

    missing = [name for name, col in [("object_id", OBJ), ("timestamp", TS), ("heading_deg", HDG), ("speed_mps", SPD)] if col is None]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}\nColumns seen: {cols}")

    usecols = [OBJ, TS, HDG, SPD] + ([TYP] if TYP else [])
    dtypes = {
        OBJ: "string",
        HDG: "float32",
        SPD: "float32",
    }
    if TYP:
        dtypes[TYP] = "category"

    tails: dict[str, TrackTail] = {}

    # Stream chunks
    for chunk in pd.read_csv(
        path,
        sep=sep,
        encoding="utf-8-sig",
        usecols=usecols,
        dtype=dtypes,
        chunksize=args.chunk_size,
        engine="c",
        low_memory=True,
    ):
        # Optional type filter
        if args.only_type.lower() != "any" and TYP and (TYP in chunk.columns):
            chunk = chunk[chunk[TYP].astype("string").str.lower() == args.only_type.lower()]
            if chunk.empty:
                continue

        # Speed filter
        chunk = chunk.dropna(subset=[OBJ, TS, SPD, HDG])
        chunk = chunk[chunk[SPD] >= args.min_speed]
        if chunk.empty:
            continue

        # Parse timestamps (after filtering to reduce cost)
        chunk[TS] = pd.to_datetime(chunk[TS], utc=True, errors="coerce")
        chunk = chunk[chunk[TS].notna()]
        if chunk.empty:
            continue

        # Normalize heading with optional offset
        chunk[HDG] = (chunk[HDG].astype("float64") + args.heading_offset) % 360.0

        # Keep only last K rows per object in THIS chunk (preserves file order)
        K = args.tail_max_samples
        tail_rows = chunk.groupby(OBJ, sort=False).tail(K)

        # Update per-object tail state (by group, not by row)
        for oid, grp in tail_rows.groupby(OBJ, sort=False):
            oid = str(oid)
            ts_last = grp[TS].iloc[-1]
            hdgs = grp[HDG].astype("float64").tolist()

            if oid not in tails:
                tails[oid] = TrackTail(last_ts=ts_last, headings=deque(maxlen=K))
            t = tails[oid]

            # Only accept updates that are as-newer-than current last_ts
            if ts_last >= t.last_ts:
                t.last_ts = ts_last
                for h in hdgs:
                    t.headings.append(float(h))

    # Decide EXIT direction per object using circular mean of last K headings
    totals = defaultdict(int)  # dir -> cars
    ts_counts = defaultdict(lambda: {"N": 0, "E": 0, "S": 0, "W": 0})  # bin -> dict(dir->count)

    for oid, t in tails.items():
        if not t.headings:
            continue
        mean_h = circular_mean_deg(list(t.headings))
        if math.isnan(mean_h):
            continue

        d = heading_to_cardinal4(mean_h)
        totals[d] += 1

        bin_ts = pd.Timestamp(t.last_ts).floor(args.interval)
        bin_key = bin_ts.strftime("%Y-%m-%d %H:%M:%S")
        ts_counts[bin_key][d] += 1

    # Build outputs
    directions = ["N", "E", "S", "W"]
    total_df = pd.DataFrame([{"direction4": d, "count": int(totals.get(d, 0))} for d in directions])

    # Timeseries DF
    ts_rows = []
    for bin_key, dmap in ts_counts.items():
        for d in directions:
            ts_rows.append({"time_bin": bin_key, "direction4": d, "count": int(dmap.get(d, 0))})
    ts_df = pd.DataFrame(ts_rows).sort_values("time_bin") if ts_rows else pd.DataFrame(columns=["time_bin","direction4","count"])

    # Write HTML + JSON
    out_total_html = outdir / "car_counts_by_exit_total.html"
    out_total_json = outdir / "car_counts_by_exit_total.json"
    out_ts_html = outdir / "car_counts_by_exit_timeseries.html"
    out_ts_json = outdir / "car_counts_by_exit_timeseries.json"

    fig_total = px.bar(
        total_df, x="direction4", y="count",
        title="Cars by EXIT Direction (unique tracks)",
        labels={"direction4": "Direction", "count": "Unique cars"},
        text="count",
    )
    fig_total.update_traces(textposition="outside", cliponaxis=False)
    fig_total.write_html(str(out_total_html), include_plotlyjs="cdn", full_html=True)
    out_total_json.write_text(total_df.to_json(orient="records"), encoding="utf-8")

    if not ts_df.empty:
        fig_ts = px.area(
            ts_df, x="time_bin", y="count", color="direction4",
            title=f"Cars by EXIT Direction over time (bin={args.interval})",
            labels={"time_bin": "Time bin", "count": "Cars", "direction4": "Direction"},
        )
        fig_ts.write_html(str(out_ts_html), include_plotlyjs="cdn", full_html=True)
        out_ts_json.write_text(ts_df.to_json(orient="records"), encoding="utf-8")
    else:
        out_ts_json.write_text("[]", encoding="utf-8")

    print("✅ Wrote:")
    print(" ", out_total_html.resolve())
    print(" ", out_total_json.resolve())
    print(" ", out_ts_html.resolve())
    print(" ", out_ts_json.resolve())

if __name__ == "__main__":
    main()
