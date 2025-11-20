#!/usr/bin/env python3
"""
clean_stats_pipeline.py

Reads your existing expanded CSV (combined_vehicle_stats_expanded.csv),
collapses duplicate mapPath-expanded rows, computes:
  - real motion (speed, accel, jerk) between frames
  - heading + heading change
  - nearest-neighbor interactions (distance, rel speed, heading diff, closing rate, ttc)

Outputs a new file: clean_stats.csv
Does NOT modify the existing expanded CSV.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import math

R_EARTH_M = 6371000.0


# ---------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2)**2 +
         np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2)
    return 2 * R_EARTH_M * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------
# Bearing (heading in degrees)
# ---------------------------------------------------------
def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = (np.cos(lat1)*np.sin(lat2) -
         np.sin(lat1)*np.cos(lat2)*np.cos(dlon))
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def main():

    print("📥 Loading combined_vehicle_stats_expanded.csv ...")
    df = pd.read_csv("combined_vehicle_stats_expanded.csv")

    print(f"➡️ Loaded {len(df)} rows")

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # -----------------------------------------------------
    # 1. Collapse duplicate mapPath-expanded rows
    # keep ONE row per (object_id, timestamp) — last row works best
    # -----------------------------------------------------
    print("🔄 Collapsing duplicate mapPath rows...")
    df.sort_values(["object_id", "timestamp"], inplace=True)
    df_unique = df.groupby(["object_id", "timestamp"]).tail(1).copy()

    print(f"➡️ Collapsed to {len(df_unique)} real frame rows")

    # -----------------------------------------------------
    # 2. Compute motion features across consecutive frames
    # -----------------------------------------------------
    print("🚗 Computing real motion features...")

    df_unique.sort_values(["object_id", "timestamp"], inplace=True)

    # previous values
    df_unique["lat_prev"] = df_unique.groupby("object_id")["lat"].shift(1)
    df_unique["lon_prev"] = df_unique.groupby("object_id")["lon"].shift(1)
    df_unique["t_prev"] = df_unique.groupby("object_id")["timestamp"].shift(1)

    # dt in seconds
    df_unique["dt"] = (df_unique["timestamp"] - df_unique["t_prev"]).dt.total_seconds()
    df_unique["dt"] = df_unique["dt"].replace(0, np.nan)

    # distance moved
    df_unique["dist_m"] = haversine(
        df_unique["lat_prev"], df_unique["lon_prev"],
        df_unique["lat"], df_unique["lon"]
    )

    # speed (m/s)
    df_unique["speed_mps_clean"] = df_unique["dist_m"] / df_unique["dt"]
    df_unique["speed_mps_clean"] = df_unique["speed_mps_clean"].fillna(0)

    # accel
    df_unique["accel_clean"] = df_unique.groupby("object_id")["speed_mps_clean"].diff().fillna(0)

    # jerk
    df_unique["jerk_clean"] = df_unique.groupby("object_id")["accel_clean"].diff().fillna(0)

    # heading
    df_unique["heading_deg_clean"] = bearing_deg(
        df_unique["lat_prev"], df_unique["lon_prev"],
        df_unique["lat"], df_unique["lon"]
    ).fillna(0)

    # heading change
    df_unique["d_heading_deg_clean"] = (
        df_unique.groupby("object_id")["heading_deg_clean"].diff().fillna(0)
    )

    # -----------------------------------------------------
    # 3. Interaction features (frame-based nearest neighbors)
    # -----------------------------------------------------
    print("🤝 Computing nearest-neighbor interaction features...")

    df_unique["nn_dist_m"] = np.nan
    df_unique["rel_speed_mps"] = np.nan
    df_unique["heading_diff_deg"] = np.nan
    df_unique["closing_rate_mps"] = np.nan
    df_unique["ttc_s"] = np.nan

    groups = df_unique.groupby(["location", "timestamp"])

    for (loc, t), g in groups:
        if len(g) <= 1:
            continue

        idx = g.index.to_numpy()
        lat_rad = np.radians(g["lat"].to_numpy())
        lon_rad = np.radians(g["lon"].to_numpy())
        pts = np.column_stack([lat_rad, lon_rad])

        # NN tree
        tree = BallTree(pts, metric="haversine")
        d_rad, nn = tree.query(pts, k=2)

        nn_idx = nn[:, 1]
        nn_dist_m = d_rad[:, 1] * R_EARTH_M

        df_unique.loc[idx, "nn_dist_m"] = nn_dist_m

        speeds = g["speed_mps_clean"].to_numpy()
        df_unique.loc[idx, "rel_speed_mps"] = speeds - speeds[nn_idx]

        hd = g["heading_deg_clean"].to_numpy()
        df_unique.loc[idx, "heading_diff_deg"] = np.abs(hd - hd[nn_idx])

        # closing rate
        vx = speeds * np.cos(np.radians(hd))
        vy = speeds * np.sin(np.radians(hd))
        rvx = vx - vx[nn_idx]
        rvy = vy - vy[nn_idx]

        dlat = lat_rad - lat_rad[nn_idx]
        dlon = lon_rad - lon_rad[nn_idx]
        dx = dlon * np.cos((lat_rad + lat_rad[nn_idx]) / 2) * R_EARTH_M
        dy = dlat * R_EARTH_M
        dist = np.hypot(dx, dy)

        ux = np.divide(dx, dist, out=np.zeros_like(dx), where=dist > 0)
        uy = np.divide(dy, dist, out=np.zeros_like(dy), where=dist > 0)

        closing = rvx * ux + rvy * uy
        # Avoid divide-by-zero
        safe_closing = np.where(np.abs(closing) < 1e-6, np.nan, closing)

        # Real TTC only when closing < 0 (approaching)
        ttc_raw = -dist / safe_closing

        # Replace invalid or positive TTC with infinity
        ttc = np.where((closing < 0) & np.isfinite(ttc_raw), ttc_raw, np.inf)


        df_unique.loc[idx, "closing_rate_mps"] = closing
        df_unique.loc[idx, "ttc_s"] = ttc

    # -----------------------------------------------------
    # 4. Save output
    # -----------------------------------------------------
    print("💾 Saving clean_stats.csv...")
    df_unique.to_csv("clean_stats.csv", index=False)
    print("🎉 Done! clean_stats.csv is ready.")


if __name__ == "__main__":
    main()
