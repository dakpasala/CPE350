#!/usr/bin/env python3
"""
car_count_by_exit_directionV3.py

Counts vehicles by EXIT direction (N/E/S/W) using FastAPI endpoint:
  GET /stats/combined?time_range=...&limit=...&location=...

Exit definition:
- For each object_id, pick the record with the latest timestamp within the fetched+filtered window.
- That record's heading_deg determines exit direction.

Outputs (default ./charts):
- car_count_exit_total.html
- car_count_exit_total.json
Optionally (if --interval is provided):
- car_count_exit_timeseries.html
- car_count_exit_timeseries.json

Dependencies:
  pip install requests pandas plotly
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import plotly.express as px
import requests


# -----------------------------
# Direction mapping
# -----------------------------
def heading_to_cardinal(h: float) -> str:
    """N=[315,360)∪[0,45), E=[45,135), S=[135,225), W=[225,315)"""
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
def ensure_outdir(path: str) -> Path:
    outdir = Path(path)
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def parse_dt(s: Optional[str]) -> Optional[pd.Timestamp]:
    if not s:
        return None
    ts = pd.to_datetime(s, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def normalize_interval(interval: Optional[str]) -> Optional[str]:
    """
    Accept both pandas 'T' and 'min' styles:
      '1T' -> '1min'
      'T'  -> 'min'
    """
    if not interval:
        return None
    s = interval.strip()
    if s == "T":
        return "min"
    if s.endswith("T") and len(s) > 1 and s[:-1].isdigit():
        return f"{s[:-1]}min"
    return s


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


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Count vehicles by EXIT direction from FastAPI endpoint.")

    ap.add_argument("--api-base", default="http://127.0.0.1:8000",
                    help="FastAPI base URL (default: http://127.0.0.1:8000)")
    ap.add_argument("--time-range", default="hour",
                    help='One of: "hour", "6hours", "12hours", "day", "week", "month" (default: hour)')
    ap.add_argument("--limit", type=int, default=200_000,
                    help="Max rows fetched from API (default: 200000)")
    ap.add_argument("--location", default=None, help="Filter by location (e.g., patterson)")

    # Filters
    ap.add_argument("--only-type", dest="only_type", default="car",
                    help="Filter detected_type (case-insensitive). Use 'any' to disable. Default: car")
    ap.add_argument("--confident-only", action="store_true",
                    help="If set, only include is_confident==True (if column exists).")
    ap.add_argument("--min-speed", type=float, default=0.0,
                    help="Minimum speed_mps for the EXIT record (default 0.0).")

    # Optional extra time slicing AFTER fetch
    ap.add_argument("--start", type=str, default=None,
                    help="Optional start datetime/date (UTC) applied after fetch.")
    ap.add_argument("--end", type=str, default=None,
                    help="Optional end datetime/date (exclusive, UTC) applied after fetch.")

    # Timeseries
    ap.add_argument("--interval", type=str, default=None,
                    help="If set (e.g. 1min, 5min, H), output a timeseries by exit time bins.")

    # Output
    ap.add_argument("--outdir", default="charts", help="Output directory (default: charts)")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds (default: 60)")

    args = ap.parse_args()

    outdir = ensure_outdir(args.outdir)
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

    # If API count == limit, likely truncated
    if n >= int(args.limit):
        print(f"⚠️ API returned count={n:,} and limit={args.limit:,}. "
              f"Data may be truncated; consider increasing --limit.")

    df = pd.DataFrame(payload["data"])
    if df.empty:
        print("⚠️ Empty dataframe after parsing API data.")
        return

    # Standardize required columns
    if "timestamp" not in df.columns or "object_id" not in df.columns or "heading_deg" not in df.columns:
        raise ValueError(f"Missing required columns. Have: {list(df.columns)}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df[df["timestamp"].notna()]
    if df.empty:
        print("⚠️ No valid timestamps.")
        return

    # Apply extra time filter (optional)
    if start is not None:
        df = df[df["timestamp"] >= start]
    if end is not None:
        df = df[df["timestamp"] < end]
    if df.empty:
        print("⚠️ No data after --start/--end filtering.")
        return

    # Type filter (optional)
    if args.only_type.lower() != "any" and "detected_type" in df.columns:
        df["detected_type"] = df["detected_type"].astype("string")
        df = df[df["detected_type"].str.lower() == args.only_type.lower()]
        if df.empty:
            print("⚠️ No data after --only-type filtering.")
            return

    # Confident filter (optional)
    if args.confident_only and "is_confident" in df.columns:
        df = df[df["is_confident"] == True]  # noqa: E712
        if df.empty:
            print("⚠️ No data after --confident-only filtering.")
            return

    # Speed field is optional; we apply min-speed only if available
    if "speed_mps" in df.columns:
        df["speed_mps"] = pd.to_numeric(df["speed_mps"], errors="coerce").fillna(0.0)
        df = df[df["speed_mps"] >= float(args.min_speed)]
        if df.empty:
            print("⚠️ No data after --min-speed filtering.")
            return

    # Exit record per object_id = latest timestamp
    df = df.sort_values("timestamp")
    df_exit = df.groupby("object_id", as_index=False).tail(1).copy()

    df_exit["heading_deg"] = pd.to_numeric(df_exit["heading_deg"], errors="coerce")
    df_exit = df_exit[df_exit["heading_deg"].notna()]
    if df_exit.empty:
        print("⚠️ No valid headings in exit records.")
        return

    df_exit["direction"] = df_exit["heading_deg"].apply(heading_to_cardinal)

    # Total counts
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
        title="Vehicle counts by EXIT direction (API)",
        labels={"direction": "Exit direction", "count": "Unique vehicles (exit)"},
    )

    out_total_html = outdir / "car_count_exit_total.html"
    out_total_json = outdir / "car_count_exit_total.json"
    fig_total.write_html(str(out_total_html), include_plotlyjs="cdn", full_html=True)
    out_total_json.write_text(total.to_json(orient="records"), encoding="utf-8")

    print("✅ Wrote:")
    print(" ", out_total_html.resolve())
    print(" ", out_total_json.resolve())
    print(f"Rows fetched: {len(df):,}; unique objects (exit): {len(df_exit):,}")

    # Timeseries
    if interval:
        df_exit["time_bin"] = df_exit["timestamp"].dt.floor(interval)
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
            title=f"Vehicle counts by EXIT direction over time (bin={interval}, API)",
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
