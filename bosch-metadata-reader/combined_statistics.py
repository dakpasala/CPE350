#!/usr/bin/env python3
"""
combined_statistics.py

Production batch feature-extraction worker.

Pipeline:
  vehicles (raw) → combined_stats (derived)

- Pulls raw vehicle docs from MongoDB in batches
- Expands mapPath
- Computes motion + interaction features
- Writes derived rows as JSON to MongoDB
- Deletes processed raw docs
"""

import argparse
import configparser
import concurrent.futures
import math
import numpy as np
import pandas as pd
import pymongo
from sklearn.neighbors import BallTree
from tqdm import tqdm
import os

# =========================
# Constants
# =========================
R_EARTH_M = 6371000.0
CONFIDENCE_THRESHOLD = 0.5
MAX_WORKERS = max(2, int((os.cpu_count() or 4) * 0.75))


# =========================
# Mongo helpers
# =========================
def _get_db():
    config = configparser.ConfigParser()
    config.read("connection.ini")
    client = pymongo.MongoClient(config["DEFAULT"]["database"])
    return client["camera-counts"]


def get_raw_collection():
    return _get_db()["vehicles"]


def get_stats_collection():
    # Mongo auto-creates if missing
    return _get_db()["combined_stats"]


# =========================
# Fetch raw batch
# =========================
def fetch_vehicle_batch(batch_size):
    coll = get_raw_collection()
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

    cursor = (
        coll.find({}, projection)
        .sort("_id", 1)
        .limit(batch_size)
    )

    return list(cursor)


# =========================
# Expand mapPath
# =========================
def expand_map_paths(data):
    rows = []

    for doc in data:
        raw_id = str(doc["_id"])
        true_id = raw_id
        ts = pd.to_datetime(doc.get("timestamp"), errors="coerce")

        if "_" in raw_id:
            maybe_id, maybe_ts = raw_id.split("_", 1)
            ts_from_id = pd.to_datetime(maybe_ts, errors="coerce")
            if not pd.isna(ts_from_id):
                true_id = maybe_id
                ts = ts_from_id

        base = {
            "object_id": true_id,
            "timestamp": ts,
            "location": doc.get("location"),
            "detected_type": doc.get("detected_type"),
            "speed": doc.get("speed"),
            "certainty": doc.get("detection_certainty"),
            "zones": doc.get("zones", []),
            "time_elapsed": doc.get("time_elapsed", 0),
        }

        path = doc.get("mapPath", [])
        if isinstance(path, list) and path:
            for i, (lat, lon) in enumerate(path):
                row = base.copy()
                row.update({"path_index": i, "lat": lat, "lon": lon})
                rows.append(row)
        else:
            row = base.copy()
            row.update({"path_index": np.nan, "lat": np.nan, "lon": np.nan})
            rows.append(row)

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in ["lat", "lon", "speed", "path_index"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# =========================
# Motion features
# =========================
def mph_to_mps(s):
    return pd.to_numeric(s, errors="coerce").fillna(0) * 0.44704


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def shortest_angle_diff_deg(a, b):
    return ((a - b + 180) % 360) - 180


def add_motion_features(df):
    df = df.sort_values(["object_id", "timestamp", "path_index"]).copy()

    df["dt_s"] = df.groupby("object_id")["timestamp"].diff().dt.total_seconds()
    df["dt_s"] = df["dt_s"].replace(0, np.nan).fillna(0.1)

    df["lat_prev"] = df.groupby("object_id")["lat"].shift(1)
    df["lon_prev"] = df.groupby("object_id")["lon"].shift(1)

    lat1 = np.radians(df["lat_prev"])
    lon1 = np.radians(df["lon_prev"])
    lat2 = np.radians(df["lat"])
    lon2 = np.radians(df["lon"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    dist_m = R_EARTH_M * c

    df["speed_mps"] = np.where(
        dist_m > 0.01,
        dist_m / df["dt_s"],
        mph_to_mps(df["speed"])
    )

    df["heading_deg"] = bearing_deg(
        df["lat_prev"].fillna(df["lat"]),
        df["lon_prev"].fillna(df["lon"]),
        df["lat"],
        df["lon"]
    )

    df["d_heading_deg"] = shortest_angle_diff_deg(
        df["heading_deg"],
        df.groupby("object_id")["heading_deg"].shift(1)
    ).fillna(0)

    df["zone_change"] = (
        df.groupby("object_id")["zones"].shift(1).astype(str) != df["zones"].astype(str)
    ).astype(int)

    df["path_gap"] = (
        df.groupby("object_id")["path_index"].diff().fillna(1) > 1
    ).astype(int)

    df["is_confident"] = df["certainty"].fillna(0) > CONFIDENCE_THRESHOLD

    accel_blocks = []
    for oid, g in df.groupby("object_id"):
        g = g.set_index("timestamp").sort_index()
        spd = g["speed_mps"].resample("1S").mean().interpolate()
        accel = spd.diff().fillna(0)
        jerk = accel.diff().fillna(0)
        g["accel"] = accel.reindex(g.index, method="nearest")
        g["jerk"] = jerk.reindex(g.index, method="nearest")
        accel_blocks.append(g.reset_index())

    return pd.concat(accel_blocks)


# =========================
# Interaction features
# =========================
def compute_interaction_for_group(args):
    g_pos, lat, lon, spd, hdg, vx, vy = args
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


def add_interaction_features(df):
    for c in ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]:
        df[c] = np.nan

    lat = np.radians(df["lat"].fillna(0).to_numpy())
    lon = np.radians(df["lon"].fillna(0).to_numpy())
    spd = df["speed_mps"].fillna(0).to_numpy()
    hdg = df["heading_deg"].fillna(0).to_numpy()

    vx = spd * np.cos(np.radians(hdg))
    vy = spd * np.sin(np.radians(hdg))

    pos = np.arange(len(df))

    groups = []
    for (_, _), g in df.groupby(["location", "timestamp"], sort=False):
        if len(g) > 1:
            groups.append((pos[g.index], lat, lon, spd, hdg, vx, vy))

    with concurrent.futures.ThreadPoolExecutor(MAX_WORKERS) as ex:
        for r in tqdm(ex.map(compute_interaction_for_group, groups), total=len(groups)):
            if r is None:
                continue
            g_pos, *vals = r
            df.iloc[g_pos, df.columns.get_indexer(
                ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]
            )] = np.column_stack(vals)

    return df


# =========================
# Save + delete
# =========================
def save_stats(df):
    coll = get_stats_collection()

    keep = [
        "object_id", "timestamp", "location", "detected_type",
        "speed_mps", "accel", "jerk", "heading_deg", "d_heading_deg",
        "nn_dist_m", "closing_rate_mps", "ttc_s", "rel_speed_mps",
        "heading_diff_deg", "zone_change", "path_gap",
        "certainty", "is_confident", "lat", "lon"
    ]

    records = df[keep].dropna(subset=["timestamp"]).to_dict("records")
    if records:
        coll.insert_many(records, ordered=False)


def delete_raw(raw_docs):
    ids = [d["_id"] for d in raw_docs]
    if ids:
        get_raw_collection().delete_many({"_id": {"$in": ids}})


# =========================
# Batch runner
# =========================
def run_batches(batch_size):
    while True:
        raw = fetch_vehicle_batch(batch_size)
        if not raw:
            print("✅ No more raw docs.")
            break

        print(f"🚚 Processing {len(raw)} raw docs")
        df = expand_map_paths(raw)
        df = add_motion_features(df)
        df = add_interaction_features(df)

        save_stats(df)
        delete_raw(raw)

        if len(raw) < batch_size:
            break


# =========================
# CLI
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=100_000)
    args = p.parse_args()
    run_batches(args.batch_size)
