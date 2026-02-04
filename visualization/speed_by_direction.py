#!/usr/bin/env python3
"""
Average SPEED by EXIT direction (N/E/S/W) using the last K heading samples per object.

Reads big CSVs safely via chunking.

Outputs (default ./charts):
- speed_by_exit_total.html / .json
- speed_by_exit_timeseries.html / .json

Example:
  python speed_by_direction.py --csv ..\\bosch-metadata-reader\\combined_vehicle_stats_with_derivatives.csv
  python speed_by_direction.py --csv your.csv --interval H --min-speed 0.5
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
# Column/header helpers
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

def mps_to_mph(x: float) -> float:
    return x * 2.2369362920544


# ---------------------------
# Track state
# ---------------------------

@dataclass
class TrackState:
    last_ts: pd.Timestamp
    tail_headings: deque  # deque[float]
    sum_speed: float
    n_speed: int


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Average speed by EXIT direction (unique tracks).")
    ap.add_argument("--csv", required=True, help="Path to CSV.")
    ap.add_argument("--outdir", default="charts", help="Output directory (default: charts)")
    ap.add_argument("--only-type", dest="only_type", default="car",
                    help="Filter detected_type (case-insensitive). Use 'any' to disable. Default: car")
    ap.add_argument("--min-speed", type=float, default=0.2,
                    help="Ignore rows with speed_mps below this (m/s). Default 0.2")
    ap.add_argument("--heading-offset", type=float, default=0.0,
                    help="Rotate heading by this many degrees. Default 0")
    ap.add_argument("--tail-max-samples", type=int, default=5,
                    help="How many last heading samples per object to keep. Default 5")
    ap.add_argument("--interval", default="H",
                    help="Timeseries bin size (H, 15T, T, D, etc). Default H")
    ap.add_argument("--chunk-size", type=int, default=500_000,
                    help="CSV chunk size. Default 500k")
    ap.add_argument("--units", choices=["mps", "mph"], default="mph",
                    help="Output units for speed. Default mph")
    ap.add_argument("--stat", choices=["mean", "median"], default="mean",
                    help="Statistic across vehicles per bin/direction. Default mean")
    args = ap.parse_args()

    path = Path(args.csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sep = sniff_delimiter(path)

    # Read header only
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
    dtypes = {OBJ: "string", HDG: "float32", SPD: "float32"}
    if TYP:
        dtypes[TYP] = "category"

    states: dict[str, TrackState] = {}
    K = args.tail_max_samples

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

        # Drop junk / NaNs early
        chunk = chunk.dropna(subset=[OBJ, TS, SPD, HDG])

        # Speed filter (noise gate)
        chunk = chunk[chunk[SPD] >= args.min_speed]
        if chunk.empty:
            continue

        # Parse timestamps (after filtering)
        chunk[TS] = pd.to_datetime(chunk[TS], utc=True, errors="coerce")
        chunk = chunk[chunk[TS].notna()]
        if chunk.empty:
            continue

        # Normalize headings
        chunk[HDG] = (chunk[HDG].astype("float64") + args.heading_offset) % 360.0

        # --- 1) Update per-object speed sums (track mean)
        # Group just the columns we need
        speed_grp = chunk.groupby(OBJ, sort=False)[SPD].agg(["sum", "count"])
        for oid, row in speed_grp.iterrows():
            oid = str(oid)
            ssum = float(row["sum"])
            ccnt = int(row["count"])

            if oid not in states:
                # We'll fill last_ts/headings in step 2
                states[oid] = TrackState(
                    last_ts=pd.Timestamp.min.tz_localize("UTC"),
                    tail_headings=deque(maxlen=K),
                    sum_speed=0.0,
                    n_speed=0,
                )
            st = states[oid]
            st.sum_speed += ssum
            st.n_speed += ccnt

        # --- 2) Update per-object exit timestamp + tail headings using last K rows per object
        tail_rows = chunk.groupby(OBJ, sort=False).tail(K)
        for oid, grp in tail_rows.groupby(OBJ, sort=False):
            oid = str(oid)
            ts_last = grp[TS].iloc[-1]
            hdgs = grp[HDG].astype("float64").tolist()

            if oid not in states:
                states[oid] = TrackState(
                    last_ts=ts_last,
                    tail_headings=deque(maxlen=K),
                    sum_speed=0.0,
                    n_speed=0,
                )

            st = states[oid]
            if ts_last >= st.last_ts:
                st.last_ts = ts_last
                for h in hdgs:
                    st.tail_headings.append(float(h))

    # Decide EXIT direction + per-vehicle mean speed
    records = []
    for oid, st in states.items():
        if st.n_speed <= 0:
            continue
        if not st.tail_headings:
            continue

        mean_heading = circular_mean_deg(list(st.tail_headings))
        if math.isnan(mean_heading):
            continue
        direction = heading_to_cardinal4(mean_heading)

        speed_mps = st.sum_speed / st.n_speed
        speed_val = mps_to_mph(speed_mps) if args.units == "mph" else speed_mps

        records.append({
            "object_id": oid,
            "exit_time": st.last_ts,
            "direction4": direction,
            "speed": speed_val,
        })

    df = pd.DataFrame(records)
    if df.empty:
        print("⚠️ No data after filters. Try lowering --min-speed or using --only-type any.")
        return

    # Total (one value per vehicle) -> summary by direction
    aggfunc = "mean" if args.stat == "mean" else "median"
    total = df.groupby("direction4", as_index=False)["speed"].agg(aggfunc)
    total = total.rename(columns={"speed": f"{aggfunc}_speed_{args.units}"})

    # Timeseries (bin by exit time)
    df["time_bin"] = pd.to_datetime(df["exit_time"], utc=True).dt.floor(args.interval)
    ts = df.groupby(["time_bin", "direction4"], as_index=False)["speed"].agg(aggfunc)
    ts = ts.rename(columns={"speed": f"{aggfunc}_speed_{args.units}"})

    # Output paths
    out_total_html = outdir / "speed_by_exit_total.html"
    out_total_json = outdir / "speed_by_exit_total.json"
    out_ts_html = outdir / "speed_by_exit_timeseries.html"
    out_ts_json = outdir / "speed_by_exit_timeseries.json"

    # Plot: total
    ycol = f"{aggfunc}_speed_{args.units}"
    fig_total = px.bar(
        total,
        x="direction4",
        y=ycol,
        title=f"{aggfunc.capitalize()} speed by EXIT direction (unique vehicles) [{args.units}]",
        labels={"direction4": "Direction", ycol: f"{aggfunc.capitalize()} speed ({args.units})"},
        text=ycol,
    )
    fig_total.update_traces(textposition="outside", cliponaxis=False)
    fig_total.write_html(str(out_total_html), include_plotlyjs="cdn", full_html=True)
    out_total_json.write_text(total.to_json(orient="records", date_format="iso"), encoding="utf-8")

    # Plot: timeseries
    fig_ts = px.line(
        ts,
        x="time_bin",
        y=ycol,
        color="direction4",
        markers=True,
        title=f"{aggfunc.capitalize()} speed by EXIT direction over time (bin={args.interval}) [{args.units}]",
        labels={"time_bin": "Exit time bin", ycol: f"{aggfunc.capitalize()} speed ({args.units})", "direction4": "Direction"},
    )
    fig_ts.write_html(str(out_ts_html), include_plotlyjs="cdn", full_html=True)
    out_ts_json.write_text(ts.to_json(orient="records", date_format="iso"), encoding="utf-8")

    print("✅ Wrote:")
    print(" ", out_total_html.resolve())
    print(" ", out_total_json.resolve())
    print(" ", out_ts_html.resolve())
    print(" ", out_ts_json.resolve())


if __name__ == "__main__":
    main()
