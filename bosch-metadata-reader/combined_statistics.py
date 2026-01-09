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
MAX_WORKERS = max(2, int((__import__("os").cpu_count() or 4) * 0.75))


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
    rows = []
    for doc in data:
        raw_id = str(doc.get("_id")).strip()

        true_id = raw_id
        ts = doc.get("timestamp")
        ts_parsed = pd.to_datetime(ts, errors="coerce")

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
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1.to_numpy(), lon1.to_numpy(), lat2.to_numpy(), lon2.to_numpy()]
    )
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0


def shortest_angle_diff_deg(a, b):
    return ((a - b + 180.0) % 360.0) - 180.0


# =========================
# Motion Features
# =========================
def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.sort_values(["object_id", "timestamp", "path_index"], inplace=True)

    df["dt_s"] = (
        df.groupby("object_id")["timestamp"]
        .diff()
        .dt.total_seconds()
    )
    df["dt_s"] = df["dt_s"].replace(0, np.nan).fillna(0.1)

    df["lat_prev"] = df.groupby("object_id")["lat"].shift(1)
    df["lon_prev"] = df.groupby("object_id")["lon"].shift(1)

    dist_m = np.zeros(len(df), dtype="float32")
    valid = (
        df["lat_prev"].notna()
        & df["lon_prev"].notna()
        & df["lat"].notna()
        & df["lon"].notna()
    )

    if valid.any():
        lat1 = np.radians(df.loc[valid, "lat_prev"])
        lon1 = np.radians(df.loc[valid, "lon_prev"])
        lat2 = np.radians(df.loc[valid, "lat"])
        lon2 = np.radians(df.loc[valid, "lon"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        dist_m[valid] = (R_EARTH_M * c).astype("float32")

    df["dist_m"] = dist_m
    df["speed_mps"] = np.where(
        df["dist_m"] > 0.01,
        df["dist_m"] / df["dt_s"],
        mph_to_mps(df["speed"])
    ).astype("float32")

    df["heading_deg"] = bearing_deg(
        df.groupby("object_id")["lat"].shift(1).fillna(df["lat"]),
        df.groupby("object_id")["lon"].shift(1).fillna(df["lon"]),
        df["lat"],
        df["lon"]
    ).astype("float32")

    df["d_heading_deg"] = shortest_angle_diff_deg(
        df["heading_deg"],
        df.groupby("object_id")["heading_deg"].shift(1)
    ).fillna(0).astype("float32")

    df["zone_change"] = (
        df.groupby("object_id")["zones"].shift(1).astype(str) != df["zones"].astype(str)
    ).astype("int8")

    df["path_gap"] = (
        df.groupby("object_id")["path_index"].diff().fillna(1) > 1
    ).astype("int8")

    df["is_confident"] = df["certainty"].fillna(0) > CONFIDENCE_THRESHOLD

    accel_blocks = []
    for oid, g in df.groupby("object_id"):
        g2 = g.set_index("timestamp").sort_index()
        speed_resampled = g2["speed_mps"].resample("1S").mean().interpolate()
        accel = speed_resampled.diff().fillna(0)
        jerk = accel.diff().fillna(0)

        g2["accel"] = accel.reindex(g2.index, method="nearest")
        g2["jerk"] = jerk.reindex(g2.index, method="nearest")
        accel_blocks.append(g2.reset_index())

    return pd.concat(accel_blocks).sort_values(["object_id", "timestamp", "path_index"])


# =========================
# Interaction Features
# =========================
def compute_interaction_for_group(args):
    loc, t, g_pos, lat, lon, spd, hdg, vx, vy = args
    if len(g_pos) <= 1:
        return None

    pts = np.column_stack([lat[g_pos], lon[g_pos]])
    tree = BallTree(pts, metric="haversine")
    d, idx = tree.query(pts, k=2)

    nn = idx[:, 1]
    dist = d[:, 1] * R_EARTH_M
    rel_speed = np.abs(spd[g_pos] - spd[g_pos][nn])
    heading_diff = np.abs(((hdg[g_pos] - hdg[g_pos][nn] + 180) % 360) - 180)

    dx = (lon[g_pos] - lon[g_pos][nn]) * np.cos(lat[g_pos]) * R_EARTH_M
    dy = (lat[g_pos] - lat[g_pos][nn]) * R_EARTH_M
    dist_xy = np.hypot(dx, dy)

    ux = np.divide(dx, dist_xy, out=np.zeros_like(dx), where=dist_xy > 0)
    uy = np.divide(dy, dist_xy, out=np.zeros_like(dy), where=dist_xy > 0)

    rvx = vx[g_pos] - vx[g_pos][nn]
    rvy = vy[g_pos] - vy[g_pos][nn]
    closing_rate = rvx * ux + rvy * uy

    ttc = np.where(closing_rate < 0, -dist_xy / closing_rate, np.inf)

    return g_pos, dist, rel_speed, heading_diff, closing_rate, ttc


def add_interaction_features_parallel(df, sample_step=1):
    df = df.copy()
    for c in ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]:
        df[c] = np.nan

    lat = np.radians(df["lat"].fillna(0).to_numpy())
    lon = np.radians(df["lon"].fillna(0).to_numpy())
    spd = df["speed_mps"].fillna(0).to_numpy()
    hdg = df["heading_deg"].fillna(0).to_numpy()

    vx = spd * np.cos(np.radians(hdg))
    vy = spd * np.sin(np.radians(hdg))

    pos = pd.Series(np.arange(len(df)), index=df.index)

    tasks = []
    for (loc, t), g in df.groupby(["location", "timestamp"], sort=False):
        if len(g) <= 1:
            continue
        if np.random.randint(0, sample_step) != 0:
            continue
        tasks.append((loc, t, pos.loc[g.index].to_numpy(), lat, lon, spd, hdg, vx, vy))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in tqdm(ex.map(compute_interaction_for_group, tasks), total=len(tasks)):
            if r is None:
                continue
            g_pos, *vals = r
            df.iloc[g_pos, df.columns.get_indexer(
                ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]
            )] = np.column_stack(vals)

    return df


# =========================
# Save
# =========================
def save_results(df):
    out = "combined_vehicle_stats_expandedNEW.csv"
    exists = os.path.exists(out)

    keep = [
        "object_id", "timestamp", "location", "detected_type",
        "speed_mps", "accel", "jerk", "heading_deg", "d_heading_deg",
        "nn_dist_m", "closing_rate_mps", "ttc_s", "rel_speed_mps",
        "heading_diff_deg", "zone_change", "path_gap",
        "certainty", "is_confident", "lat", "lon"
    ]

    df[keep].dropna(how="all").to_csv(
        out, mode="a", header=not exists, index=False
    )

    print(f"💾 {'Appended to' if exists else 'Created'} → {out}")


# =========================
# API-SAFE PIPELINE (NEW)
# =========================
def run_feature_pipeline(limit=None, interactions=True, sample_step=1):
    data = fetch_vehicle_data(limit)
    df = expand_map_paths(data)
    df = add_motion_features(df)

    if interactions:
        df = add_interaction_features_parallel(df, sample_step)
    else:
        for c in ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]:
            df[c] = np.nan

    save_results(df)
    return df


# =========================
# CLI ENTRYPOINT (UNCHANGED)
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-interactions", action="store_true")
    p.add_argument("--sample-step", type=int, default=1)
    args = p.parse_args()

    run_feature_pipeline(
        limit=args.limit,
        interactions=not args.no_interactions,
        sample_step=args.sample_step
    )
