#!/usr/bin/env python3
"""
Car counts by direction (N/E/S/W), robust to BOMs, delimiters, and column name variants.
Chunk-friendly for very large CSVs.

Outputs (to charts/ by default):
- car_counts_by_direction_total.html / .json
- car_counts_by_direction_timeseries.html / .json
"""

from __future__ import annotations
import argparse
from pathlib import Path
import json
import csv
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import plotly.express as px

# ---------------- helpers ----------------

def norm(s: str) -> str:
    """Normalize a column name for matching: lowercase, strip BOM/whitespace, remove non-alnum."""
    if s is None:
        return ""
    s = str(s).lstrip("\ufeff").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "", s)  # collapse underscores/spaces/punct
    return s

def pick_column(cols: list[str], candidates: list[str]) -> str | None:
    """Pick the first matching column name (by normalized form) from a list of candidates."""
    norm_map = {norm(c): c for c in cols}  # normalized -> original
    for cand in candidates:
        if norm(cand) in norm_map:
            return norm_map[norm(cand)]
    # also try partials (e.g., "objectid" within "object_id_")
    for c in cols:
        if any(norm(cand) == norm(c) for cand in candidates):
            return c
    return None

def sniff_delimiter(path: Path) -> str:
    """Detect CSV delimiter among common ones."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        sample = fh.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
        return dialect.delimiter
    except Exception:
        # fallback: if we see many semicolons/tabs, use that; else comma
        counts = {d: sample.count(d) for d in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get)

def heading_to_cardinal4_array(h: pd.Series) -> pd.Series:
    """Vectorized mapping of heading degrees (0=North) to N/E/S/W."""
    h = (h.astype("float32") % 360.0)
    n = (h >= 315) | (h < 45)
    e = (h >= 45) & (h < 135)
    s = (h >= 135) & (h < 225)
    out = pd.Series(
        np.where(n, "N", np.where(e, "E", np.where(s, "S", "W"))),
        index=h.index,
        dtype="category"
    )
    return out

def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser(description="Count unique cars by direction (and by time bin) from a large CSV.")
    ap.add_argument("--csv", required=True, help="Path to CSV.")
    ap.add_argument("--outdir", default="charts", help="Output directory (default: charts)")
    ap.add_argument("--interval", default="H",
                    help="Time bin size for timeseries (e.g., H=hour, 15T=15 min, D=day). Default H")
    ap.add_argument("--min-speed", type=float, default=0.2,
                    help="Ignore rows below this speed (m/s). Default 0.2")
    ap.add_argument("--only-type", type=str, default="car",
                    help="Filter by detected_type (case-insensitive). Use 'all' to keep all.")
    ap.add_argument("--heading-offset", type=float, default=0.0,
                    help="Rotate heading by this many degrees if your convention differs. Default 0")
    ap.add_argument("--chunk-size", type=int, default=500_000, help="CSV chunk size. Default 500k")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # --- Detect delimiter and read header ---
    sep = sniff_delimiter(csv_path)
    header_df = pd.read_csv(csv_path, nrows=0, sep=sep, encoding="utf-8-sig")
    cols = list(header_df.columns)

    # --- Resolve required & optional columns by synonyms ---
    OBJ_COL = pick_column(cols, ["object_id", "objectid", "id", "track_id", "trackid", "uuid"])
    TIME_COL = pick_column(cols, ["timestamp", "time", "utc_time", "utctime", "datetime", "date_time", "ts"])
    HEAD_COL = pick_column(cols, ["heading_deg", "heading", "bearing_deg", "bearing", "compass"])
    SPEED_COL = pick_column(cols, ["speed_mps", "speed", "speedms", "speed_m_s", "velocity", "velocity_mps"])
    TYPE_COL  = pick_column(cols, ["detected_type", "type", "class", "label", "obj_type"])

    missing = [name for name, col in [("object_id", OBJ_COL), ("timestamp", TIME_COL), ("heading_deg", HEAD_COL)] if col is None]
    if missing:
        raise ValueError(
            "CSV missing required columns (after normalization). "
            f"Could not find: {missing}\n"
            f"Columns seen: {cols}"
        )

    usecols = [OBJ_COL, TIME_COL, HEAD_COL]
    if SPEED_COL: usecols.append(SPEED_COL)
    if TYPE_COL:  usecols.append(TYPE_COL)

    dtypes = {
        OBJ_COL: "string",
        HEAD_COL: "float32",
    }
    if SPEED_COL: dtypes[SPEED_COL] = "float32"
    if TYPE_COL:  dtypes[TYPE_COL]  = "category"

    # --- Accumulators ---
    directions = ["N", "E", "S", "W"]
    total_counts = {d: 0 for d in directions}
    timeseries_counts: dict[str, dict[str, int]] = defaultdict(lambda: {d: 0 for d in directions})

    seen_objects = set()     # object_ids counted once overall
    seen_obj_bin = set()     # "object_id|bin" pairs counted in timeseries

    # --- Stream in chunks ---
    for chunk in pd.read_csv(
        csv_path,
        usecols=usecols,
        dtype=dtypes,
        sep=sep,
        chunksize=args.chunk_size,
        encoding="utf-8-sig",
        engine="c",
        low_memory=True
    ):
        # Drop exact dupes within chunk
        chunk = chunk.drop_duplicates(subset=[OBJ_COL, TIME_COL])

        # Filter by detected_type if requested
        if args.only_type.lower() != "all" and TYPE_COL in chunk.columns:
            mask = chunk[TYPE_COL].astype("string").str.lower() == args.only_type.lower()
            chunk = chunk[mask]

        if chunk.empty:
            continue

        # Filter by speed if available
        if SPEED_COL in chunk.columns:
            chunk = chunk[chunk[SPEED_COL] >= args.min_speed]
            if chunk.empty:
                continue

        # Heading → direction
        h = (chunk[HEAD_COL] + args.heading_offset) % 360.0
        chunk["direction4"] = heading_to_cardinal4_array(h)

        # Parse timestamps and create time bins
        chunk[TIME_COL] = pd.to_datetime(chunk[TIME_COL], utc=True, errors="coerce")
        chunk = chunk[chunk[TIME_COL].notna()]
        if chunk.empty:
            continue

        time_bin = chunk[TIME_COL].dt.floor(args.interval)
        chunk["bin_iso"] = time_bin.dt.strftime("%Y-%m-%d %H:%M:%S")

        # ---- TOTAL unique cars per direction (count each object once overall) ----
        new_mask = ~chunk[OBJ_COL].isin(seen_objects)
        if new_mask.any():
            first_rows = chunk.loc[new_mask, [OBJ_COL, "direction4"]].drop_duplicates(OBJ_COL, keep="first")
            by_dir = first_rows.groupby("direction4")[OBJ_COL].nunique()
            for d, cnt in by_dir.items():
                total_counts[str(d)] += int(cnt)
            seen_objects.update(first_rows[OBJ_COL].astype(str).tolist())

        # ---- TIMESERIES unique per bin per direction ----
        first_bin = chunk[[OBJ_COL, "bin_iso", "direction4"]].drop_duplicates([OBJ_COL, "bin_iso"], keep="first")
        keys = (first_bin[OBJ_COL].astype(str) + "|" + first_bin["bin_iso"].astype(str))
        new_pairs_mask = ~keys.isin(seen_obj_bin)
        new_pairs = first_bin.loc[new_pairs_mask]
        if not new_pairs.empty:
            grp = new_pairs.groupby(["bin_iso", "direction4"]).size()
            for (bin_iso, d), cnt in grp.items():
                timeseries_counts[str(bin_iso)][str(d)] += int(cnt)
            seen_obj_bin.update((new_pairs[OBJ_COL].astype(str) + "|" + new_pairs["bin_iso"].astype(str)).tolist())

    # --- Build outputs ---

    # Totals
    total_rows = [{"direction4": d, "count": total_counts[d]} for d in directions]
    total_df = pd.DataFrame(total_rows)

    # Timeseries
    ts_rows = []
    for bin_iso, dmap in timeseries_counts.items():
        for d in directions:
            ts_rows.append({"time_bin": bin_iso, "direction4": d, "count": dmap.get(d, 0)})
    ts_df = pd.DataFrame(ts_rows)
    if not ts_df.empty:
        ts_df = ts_df.sort_values("time_bin")

    # JSON
    write_json(outdir / "car_counts_by_direction_total.json", total_rows)
    write_json(outdir / "car_counts_by_direction_timeseries.json", ts_rows)

    # Charts
    fig_total = px.bar(
        total_df, x="direction4", y="count",
        title="Total Unique Cars by Direction",
        labels={"direction4": "Direction", "count": "Unique cars"},
        text="count"
    )
    fig_total.update_traces(textposition="outside", cliponaxis=False)
    fig_total.write_html(str(outdir / "car_counts_by_direction_total.html"),
                         include_plotlyjs="cdn", full_html=True)

    if not ts_df.empty:
        fig_ts = px.area(
            ts_df, x="time_bin", y="count", color="direction4",
            title=f"Unique Cars by Direction per Time Bin",
            labels={"time_bin": "Time", "count": "Unique cars", "direction4": "Direction"}
        )
        fig_ts.write_html(str(outdir / "car_counts_by_direction_timeseries.html"),
                          include_plotlyjs="cdn", full_html=True)

    print("\n✅ Wrote:")
    print("  -", (outdir / "car_counts_by_direction_total.html").resolve())
    if not ts_df.empty:
        print("  -", (outdir / "car_counts_by_direction_timeseries.html").resolve())
    print("  -", (outdir / "car_counts_by_direction_total.json").resolve())
    print("  -", (outdir / "car_counts_by_direction_timeseries.json").resolve())

if __name__ == "__main__":
    main()
