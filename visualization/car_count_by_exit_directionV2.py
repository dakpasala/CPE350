#!/usr/bin/env python3
"""
car_count_by_exit_direction.py

Counts vehicles by EXIT direction (N/E/S/W) using MongoDB data.

Definition:
- For each object_id, we pick the record with the latest timestamp in the selected time window.
- That record's heading_deg determines the "exit direction".
- That record's timestamp determines the time bin (if --interval is used).

Outputs (default ./charts):
- car_count_exit_total.html
- car_count_exit_total.json
Optionally (if --interval is provided):
- car_count_exit_timeseries.html
- car_count_exit_timeseries.json

Requires:
  pip install pymongo python-dotenv dnspython pandas plotly
"""

from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

import pandas as pd
import plotly.express as px
from dotenv import load_dotenv, find_dotenv
from pymongo import MongoClient


# -----------------------------
# Direction mapping
# -----------------------------
def heading_to_cardinal(h: float) -> str:
    """Map heading degrees to N/E/S/W using:
       N=[315,360)∪[0,45), E=[45,135), S=[135,225), W=[225,315)"""
    h = float(h) % 360.0
    if h >= 315.0 or h < 45.0:
        return "N"
    if h < 135.0:
        return "E"
    if h < 225.0:
        return "S"
    return "W"


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


def ensure_outdir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def mongo_client_from_env() -> Tuple[MongoClient, str, str]:
    load_dotenv(find_dotenv())
    uri = os.getenv("MONGO_URI")
    dbn = os.getenv("MONGO_DB", "camera-counts")
    col = os.getenv("MONGO_COLL", "combined_stats")

    if not uri:
        raise RuntimeError("Missing MONGO_URI in your .env file.")
    return MongoClient(uri), dbn, col


def build_query(
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
    only_type: str,
    confident_only: bool,
) -> Dict[str, Any]:
    q: Dict[str, Any] = {}

    # time window (assumes stored as datetime or ISO string; Mongo can compare datetime properly)
    # We'll still filter again in pandas as a safety net.
    if start or end:
        q["timestamp"] = {}
        if start is not None:
            q["timestamp"]["$gte"] = start.to_pydatetime()
        if end is not None:
            q["timestamp"]["$lt"] = end.to_pydatetime()

    if only_type.lower() != "any":
        q["detected_type"] = {"$regex": f"^{only_type}$", "$options": "i"}

    if confident_only:
        q["is_confident"] = True

    return q


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Count vehicles by EXIT direction from MongoDB.")
    ap.add_argument("--outdir", default="charts", help="Output directory (default: charts)")
    ap.add_argument("--only-type", dest="only_type", default="car",
                    help="Filter detected_type (case-insensitive). Use 'any' to disable. Default: car")
    ap.add_argument("--confident-only", action="store_true",
                    help="If set, only include rows where is_confident==True (if the field exists).")
    ap.add_argument("--start", type=str, default=None, help="Start datetime/date (UTC recommended).")
    ap.add_argument("--end", type=str, default=None, help="End datetime/date (exclusive).")
    ap.add_argument("--interval", type=str, default=None,
                    help="If set (e.g. H, 15T), output a timeseries by EXIT time bins.")
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="Minimum speed_mps for the EXIT record (default 0.0).")
    ap.add_argument("--limit", type=int, default=0,
                    help="Optional: limit number of rows read from Mongo (debug). 0 = no limit.")
    ap.add_argument("--batch-size", type=int, default=50_000,
                    help="Mongo cursor batch size (default 50000).")
    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)

    start = parse_dt(args.start)
    end = parse_dt(args.end)

    client, dbn, coln = mongo_client_from_env()
    coll = client[dbn][coln]

    query = build_query(start, end, args.only_type, args.confident_only)

    projection = {
        "_id": 0,
        "object_id": 1,
        "timestamp": 1,
        "detected_type": 1,
        "heading_deg": 1,
        "speed_mps": 1,
        "is_confident": 1,
    }

    cursor = coll.find(query, projection=projection, batch_size=args.batch_size)
    # Sorting makes "latest per object" stable and reduces overwrites, but may be slow without indexes.
    # If you have indexes, consider:
    # cursor = cursor.sort([("object_id", 1), ("timestamp", 1)])
    if args.limit and args.limit > 0:
        cursor = cursor.limit(args.limit)

    # Keep only latest record per object_id
    latest: Dict[str, Tuple[pd.Timestamp, float, float]] = {}
    # object_id -> (ts, heading_deg, speed_mps)

    rows_seen = 0
    rows_used = 0

    for doc in cursor:
        rows_seen += 1

        oid = doc.get("object_id")
        if not oid:
            continue

        ts = pd.to_datetime(doc.get("timestamp"), utc=True, errors="coerce")
        if pd.isna(ts):
            continue

        # safety net time filter (in case timestamp is stored as string in Mongo)
        if start is not None and ts < start:
            continue
        if end is not None and ts >= end:
            continue

        heading = doc.get("heading_deg")
        if heading is None:
            continue

        spd = doc.get("speed_mps")
        try:
            spd_f = float(spd) if spd is not None else 0.0
        except Exception:
            spd_f = 0.0

        # apply min speed on EXIT record candidate
        if spd_f < float(args.min_speed):
            continue

        try:
            heading_f = float(heading)
        except Exception:
            continue

        prev = latest.get(str(oid))
        if prev is None or ts > prev[0]:
            latest[str(oid)] = (ts, heading_f, spd_f)
            rows_used += 1

    if not latest:
        print("⚠️ No exit records found after filtering.")
        return

    # Build exit dataframe (1 row per object)
    exit_rows = []
    for oid, (ts, heading, spd) in latest.items():
        exit_rows.append({
            "object_id": oid,
            "exit_time": ts,
            "heading_deg": heading % 360.0,
            "direction": heading_to_cardinal(heading),
            "speed_mps": spd,
        })

    df_exit = pd.DataFrame(exit_rows).sort_values("exit_time")

    # ---- Total counts
    total = (
        df_exit.groupby("direction", as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("direction")
    )

    fig_total = px.bar(
        total,
        x="direction",
        y="count",
        title="Vehicle counts by EXIT direction",
        labels={"direction": "Exit direction", "count": "Unique vehicles (exit)"},
    )

    out_total_html = outdir / "car_count_exit_total.html"
    out_total_json = outdir / "car_count_exit_total.json"
    fig_total.write_html(str(out_total_html), include_plotlyjs="cdn", full_html=True)
    out_total_json.write_text(total.to_json(orient="records"), encoding="utf-8")

    print("✅ Wrote:")
    print(" ", out_total_html.resolve())
    print(" ", out_total_json.resolve())
    print(f"Rows seen: {rows_seen:,}; unique objects (exit): {len(df_exit):,}")

    # ---- Timeseries (optional)
    if args.interval:
        df_exit["time_bin"] = df_exit["exit_time"].dt.floor(args.interval)
        ts_counts = (
            df_exit.groupby(["time_bin", "direction"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("time_bin")
        )

        fig_ts = px.line(
            ts_counts,
            x="time_bin",
            y="count",
            color="direction",
            title=f"Vehicle counts by EXIT direction over time (bin={args.interval})",
            labels={"time_bin": "Exit time bin", "count": "Unique vehicles (exit)"},
        )
        fig_ts.update_layout(hovermode="x unified")

        out_ts_html = outdir / "car_count_exit_timeseries.html"
        out_ts_json = outdir / "car_count_exit_timeseries.json"
        fig_ts.write_html(str(out_ts_html), include_plotlyjs="cdn", full_html=True)
        out_ts_json.write_text(ts_counts.to_json(orient="records", date_format="iso"), encoding="utf-8")

        print(" ", out_ts_html.resolve())
        print(" ", out_ts_json.resolve())


if __name__ == "__main__":
    main()
