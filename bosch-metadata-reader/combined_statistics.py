#!/usr/bin/env python3
"""
combined_statistics.py
Final optimized multithreaded version.
Pulls from MongoDB, expands mapPath, computes:
  - Motion features (speed_mps, accel, jerk, heading, etc.)
  - Interaction features (nn_dist_m, rel_speed_mps, heading_diff_deg,
    closing_rate_mps, ttc_s)
Uses ThreadPoolExecutor for parallel nearest-neighbor processing.
"""

import argparse
import configparser
import math
import os
import concurrent.futures

import numpy as np
import pandas as pd
import pymongo
from sklearn.neighbors import BallTree
from tqdm import tqdm

# =========================
# Constants
# =========================
R_EARTH_M = 6371000.0
CONFIDENCE_THRESHOLD = 0.5
MAX_WORKERS = max(2, int((__import__("os").cpu_count() or 4) * 0.75))  # use ~75% of cores


# =========================
# Mongo connection
# =========================
def get_collection():
    config = configparser.ConfigParser()
    config.read("connection.ini")
    dbUrl = config["DEFAULT"]["database"]
    client = pymongo.MongoClient(dbUrl)
    db = client["camera-counts"]
    return db["vehicles"]


# =========================
# Fetch data
# =========================
def fetch_vehicle_data(limit=None):
    coll = get_collection()
    projection = {
        "_id": 1,
        "timestamp": 1,
        "location": 1,
        "detected_type": 1,
        "speed": 1,
        "detection_certainty": 1,
        "zones": 1,
        "time_elapsed": 1,
        "mapPath": 1,
    }
    cursor = coll.find({}, projection)
    if limit:
        cursor = cursor.limit(int(limit))
    data = list(cursor)
    print(f"✅ Retrieved {len(data)} vehicle records.")
    return data


# =========================
# Expand mapPath
# =========================
def expand_map_paths(data):
    """
    Expand each document's mapPath into per-point rows.

    IMPORTANT FIX:
    - MongoDB _id looks like "151759_2025-10-06T18:55:21.993000+00:00"
    - We split that into:
        object_id = "151759"
        timestamp = parsed("2025-10-06T18:55:21.993000+00:00")
    So all frames for the same vehicle share the same object_id.
    """
    rows = []
    for doc in data:
        raw_id = str(doc.get("_id")).strip()

        true_id = raw_id
        ts = doc.get("timestamp")
        ts_parsed = pd.to_datetime(ts, errors="coerce")

        # If _id has embedded timestamp, prefer that
        if "_" in raw_id:
            maybe_id, maybe_ts = raw_id.split("_", 1)
            ts_from_id = pd.to_datetime(maybe_ts, errors="coerce")
            if not pd.isna(ts_from_id):
                true_id = maybe_id
                ts_parsed = ts_from_id

        base = {
            "object_id": true_id,
            "timestamp": ts_parsed,
            "location": doc.get("location"),
            "detected_type": doc.get("detected_type"),
            "speed": doc.get("speed"),
            "certainty": doc.get("detection_certainty"),
            "zones": doc.get("zones", []),
            "time_elapsed": doc.get("time_elapsed", 0),
        }

        path = doc.get("mapPath", [])
        if isinstance(path, list) and path:
            for i, coords in enumerate(path):
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    lat, lon = coords
                    row = base.copy()
                    row["path_index"] = i
                    row["lat"], row["lon"] = lat, lon
                    rows.append(row)
        else:
            row = base.copy()
            row.update({"path_index": np.nan, "lat": np.nan, "lon": np.nan})
            rows.append(row)

    df = pd.DataFrame(rows)

    # enforce numeric types
    for c in ["lat", "lon", "speed"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["path_index"] = pd.to_numeric(df["path_index"], errors="coerce").astype("float32")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print(f"🧩 Expanded → {len(df)} rows.")
    return df


# =========================
# Helpers
# =========================
def mph_to_mps(s):
    return pd.to_numeric(s, errors="coerce").fillna(0) * 0.44704


def bearing_deg(lat1, lon1, lat2, lon2):
    """
    Bearing from (lat1, lon1) -> (lat2, lon2) in degrees [0, 360).
    Accepts pandas Series or numpy arrays.
    """
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1.to_numpy(), lon1.to_numpy(), lat2.to_numpy(), lon2.to_numpy()]
    )
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def shortest_angle_diff_deg(a, b):
    """
    Smallest signed difference (a - b) in degrees in [-180, 180).
    """
    return ((a - b + 180.0) % 360.0) - 180.0


# =========================
# Motion Features (with 1-second accel model)
# =========================
def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Motion pipeline:
      - Sorts by (object_id, timestamp, path_index)
      - Computes speed_mps from haversine distance / real dt
      - Falls back to Bosch speed when no movement detected
      - Computes heading + heading change
      - Computes acceleration & jerk over ~1-second windows per object
        (recommended, physically realistic model)
      - Adds zone_change, path_gap, is_confident
    """
    df = df.copy()

    # 1) Sort tracks
    df.sort_values(["object_id", "timestamp", "path_index"], inplace=True)

    # 2) Ensure types
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    # 3) Per-object time delta (seconds)
    df["dt_s"] = (
        df.groupby("object_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # For first samples or bad diffs, give a small nonzero dt just to avoid zero-div
    df["dt_s"] = df["dt_s"].replace(0, np.nan).fillna(0.1)

    # 4) Previous lat/lon per object
    df["lat_prev"] = df.groupby("object_id")["lat"].shift(1)
    df["lon_prev"] = df.groupby("object_id")["lon"].shift(1)

    # 5) Haversine distance between consecutive points (meters)
    dist_m = np.zeros(len(df), dtype="float32")
    valid_mask = (
        df["lat_prev"].notna()
        & df["lon_prev"].notna()
        & df["lat"].notna()
        & df["lon"].notna()
    )
    if valid_mask.any():
        lat1 = np.radians(df.loc[valid_mask, "lat_prev"].to_numpy())
        lon1 = np.radians(df.loc[valid_mask, "lon_prev"].to_numpy())
        lat2 = np.radians(df.loc[valid_mask, "lat"].to_numpy())
        lon2 = np.radians(df.loc[valid_mask, "lon"].to_numpy())

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2.0 * np.arcsin(np.sqrt(a))
        dist_m[valid_mask.to_numpy()] = (R_EARTH_M * c).astype("float32")

    df["dist_m"] = dist_m

    # 6) Speed (m/s) from movement
    dt = df["dt_s"].to_numpy()
    speed_from_move = np.zeros(len(df), dtype="float32")
    np.divide(dist_m, dt, out=speed_from_move, where=(dt > 0))

    # 7) Fallback to Bosch speed when no movement
    speed_fallback = mph_to_mps(df["speed"]).to_numpy(dtype="float32")
    df["speed_mps"] = np.where(speed_from_move > 0.01, speed_from_move, speed_fallback).astype(
        "float32"
    )

    # 8) Heading + heading change
    lat_prev = df.groupby("object_id")["lat"].shift(1)
    lon_prev = df.groupby("object_id")["lon"].shift(1)
    heading_vals = bearing_deg(
        lat_prev.fillna(df["lat"]),
        lon_prev.fillna(df["lon"]),
        df["lat"],
        df["lon"]
    )

    df["heading_deg"] = pd.Series(heading_vals, index=df.index).fillna(0.0).astype("float32")

    prev_heading = df.groupby("object_id")["heading_deg"].shift(1)
    df["d_heading_deg"] = (
        shortest_angle_diff_deg(df["heading_deg"], prev_heading)
        .fillna(0.0)
        .astype("float32")
    )

    # 9) zone change + path gap
    prev_z = df.groupby("object_id")["zones"].shift(1)
    df["zone_change"] = (prev_z.astype(str) != df["zones"].astype(str)).astype("int8")
    df["path_gap"] = (df.groupby("object_id")["path_index"].diff().fillna(1) > 1).astype("int8")

    # 10) confidence mask
    df["is_confident"] = df["certainty"].fillna(0.0) > CONFIDENCE_THRESHOLD

    # 11) Realistic acceleration model (1-second window per object)
    accel_blocks = []
    for oid, g in df.groupby("object_id"):
        g2 = g.sort_values(["timestamp", "path_index"]).set_index("timestamp")

        # Resample speed at 1-second intervals
        if len(g2) == 0 or g2.index.isna().all():
            # edge case: no valid timestamps
            g2["accel"] = 0.0
            g2["jerk"] = 0.0
            accel_blocks.append(g2.reset_index())
            continue

        speed_resampled = g2["speed_mps"].resample("1S").mean().interpolate()

        # Acceleration = Δv over 1 second (units m/s^2)
        accel = speed_resampled.diff().fillna(0.0)

        # Jerk = Δ(accel) over 1 second (units m/s^3)
        jerk = accel.diff().fillna(0.0)

        # Map back to original timestamps: nearest 1-second sample
        accel_at_orig = accel.reindex(g2.index, method="nearest")
        jerk_at_orig = jerk.reindex(g2.index, method="nearest")

        g2["accel"] = accel_at_orig.to_numpy(dtype="float32")
        g2["jerk"] = jerk_at_orig.to_numpy(dtype="float32")

        accel_blocks.append(g2.reset_index())  # bring timestamp back as a column

    df = pd.concat(accel_blocks, ignore_index=True)
    df.sort_values(["object_id", "timestamp", "path_index"], inplace=True)

    return df


# =========================
# Interaction Features (Parallel)
# =========================
def compute_interaction_for_group(args):
    loc, t, g_pos, lat, lon, spd, hdg, vx, vy = args
    if len(g_pos) <= 1:
        return None

    lat_rad, lon_rad = lat[g_pos], lon[g_pos]
    pts = np.column_stack([lat_rad, lon_rad])
    tree = BallTree(pts, metric="haversine")
    d_rad, nbr_idx = tree.query(pts, k=2)
    nn_local = nbr_idx[:, 1]
    nn_dist = (d_rad[:, 1] * R_EARTH_M).astype("float32")

    sp_i = spd[g_pos]
    rel_speed = np.abs(sp_i - sp_i[nn_local]).astype("float32")

    hd_i = hdg[g_pos]
    heading_diff = np.abs(((hd_i - hd_i[nn_local] + 180.0) % 360.0) - 180.0).astype("float32")

    dlat = lat_rad - lat_rad[nn_local]
    dlon = lon_rad - lon_rad[nn_local]
    dx = dlon * np.cos((lat_rad + lat_rad[nn_local]) / 2.0) * R_EARTH_M
    dy = dlat * R_EARTH_M
    dist = np.hypot(dx, dy).astype("float32")

    ux = np.divide(dx, dist, out=np.zeros_like(dist), where=dist > 0)
    uy = np.divide(dy, dist, out=np.zeros_like(dist), where=dist > 0)

    rvx = (vx[g_pos] - vx[g_pos][nn_local]).astype("float32")
    rvy = (vy[g_pos] - vy[g_pos][nn_local]).astype("float32")
    closing_rate = (rvx * ux + rvy * uy).astype("float32")

    with np.errstate(divide="ignore", invalid="ignore"):
        ttc = np.where(closing_rate < 0, -dist / closing_rate, np.inf).astype("float32")

    return (g_pos, nn_dist, rel_speed, heading_diff, closing_rate, ttc)


def add_interaction_features_parallel(df, sample_step=1):
    df = df.copy()
    n = len(df)
    for c in ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]:
        df[c] = np.nan

    lat = np.radians(df["lat"].astype(float).fillna(0.0).to_numpy())
    lon = np.radians(df["lon"].astype(float).fillna(0.0).to_numpy())
    spd = df["speed_mps"].astype(float).fillna(0.0).to_numpy()
    hdg = df["heading_deg"].astype(float).fillna(0.0).to_numpy()
    vx, vy = spd * np.cos(np.radians(hdg)), spd * np.sin(np.radians(hdg))

    pos_of_label = pd.Series(np.arange(n), index=df.index)
    group_iter = df.groupby(["location", "timestamp"], sort=False)

    tasks = []
    for (loc, t), g in group_iter:
        if len(g) <= 1:
            continue
        if np.random.randint(0, sample_step) != 0:
            continue
        g_pos = pos_of_label.loc[g.index].to_numpy()
        tasks.append((loc, t, g_pos, lat, lon, spd, hdg, vx, vy))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for result in tqdm(ex.map(compute_interaction_for_group, tasks), total=len(tasks), desc="Interactions"):
            if result is None:
                continue
            g_pos, nn_dist, rel_speed, heading_diff, closing_rate, ttc = result
            df.iloc[g_pos, df.columns.get_loc("nn_dist_m")] = nn_dist
            df.iloc[g_pos, df.columns.get_loc("rel_speed_mps")] = rel_speed
            df.iloc[g_pos, df.columns.get_loc("heading_diff_deg")] = heading_diff
            df.iloc[g_pos, df.columns.get_loc("closing_rate_mps")] = closing_rate
            df.iloc[g_pos, df.columns.get_loc("ttc_s")] = ttc

    return df


# =========================
# Save
# =========================
def save_results(df):
    keep = [
        "object_id",
        "timestamp",
        "location",
        "detected_type",
        "speed_mps",
        "accel",
        "jerk",
        "heading_deg",
        "d_heading_deg",
        "nn_dist_m",
        "closing_rate_mps",
        "ttc_s",
        "rel_speed_mps",
        "heading_diff_deg",
        "zone_change",
        "path_gap",
        "certainty",
        "is_confident",
        "lat",
        "lon",
    ]

    out_path = "combined_vehicle_stats_expandedNEW.csv"
    file_exists = os.path.exists(out_path)

    df = df[keep].dropna(how="all")

    df.to_csv(
        out_path,
        mode="a",  # append
        header=not file_exists,
        index=False,
    )

    print(f"💾 {'Appended to' if file_exists else 'Created'} → {out_path}")


# =========================
# Main
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-interactions", action="store_true")
    p.add_argument("--sample-step", type=int, default=1)
    args = p.parse_args()

    data = fetch_vehicle_data(limit=args.limit)
    df = expand_map_paths(data)
    df = add_motion_features(df)

    if not args.no_interactions:
        df = add_interaction_features_parallel(df, sample_step=args.sample_step)
    else:
        for c in ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]:
            df[c] = np.nan

    save_results(df)
