#!/usr/bin/env python3
"""
compute_motion_derivatives.py
Post-processes combined_vehicle_stats_expanded.csv to recompute accurate
motion derivatives (acceleration, jerk) based on real timestamps.

Usage:
    python3 compute_motion_derivatives.py [--input combined_vehicle_stats_expanded.csv]
"""

import argparse
import pandas as pd
import numpy as np
import os


def compute_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Compute acceleration and jerk per object using timestamp deltas."""
    df = df.copy()
    if "timestamp" not in df.columns or "speed_mps" not in df.columns:
        raise ValueError("CSV missing 'timestamp' or 'speed_mps' columns.")

    # ensure proper time format
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values(["object_id", "timestamp"], inplace=True)

    # group-based time and speed differences
    df["dt_s"] = (
        df.groupby("object_id")["timestamp"].diff().dt.total_seconds().fillna(0.1)
    )
    df["dv"] = df.groupby("object_id")["speed_mps"].diff().fillna(0.0)

    # recompute acceleration and jerk
    df["accel_new"] = (df["dv"] / df["dt_s"].replace(0, np.nan)).fillna(0.0)
    df["jerk_new"] = (
        df.groupby("object_id")["accel_new"].diff() / df["dt_s"].replace(0, np.nan)
    ).fillna(0.0)

    # optionally sanity clip unreasonable outliers
    df["accel_new"] = df["accel_new"].clip(-10, 10)
    df["jerk_new"] = df["jerk_new"].clip(-20, 20)

    print(f"✅ Recomputed derivatives for {df['object_id'].nunique()} unique objects.")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="combined_vehicle_stats_expanded.csv")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    print(f"📂 Loading {args.input} ...")
    df = pd.read_csv(args.input)

    df2 = compute_derivatives(df)

    # preserve everything + add new columns
    output = "combined_vehicle_stats_with_derivatives.csv"
    df2.to_csv(output, index=False)
    print(f"💾 Saved enriched file → {output}")
    print(f"📊 Total rows: {len(df2)}  |  Columns: {len(df2.columns)}")


if __name__ == "__main__":
    main()
