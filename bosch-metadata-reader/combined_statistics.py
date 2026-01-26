#!/usr/bin/env python3
"""
combined_statistics.py

Production batch + API feature-extraction worker.

Supports:
- Mongo batch ingestion (legacy)
- API JSON ingestion (new)

Pipeline:
  raw vehicles → combined_stats (derived)
"""

import argparse
import configparser
import concurrent.futures
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

MIN_POINTS_PER_OBJECT = 3      # ⭐ Used only for batch processing
TIME_BUCKET = "1S"             # ⭐ timestamp alignment for livestream


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
    return _get_db()["combined_stats"]


# =========================
# Fetch raw batch (LEGACY)
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
    return list(coll.find({}, projection).limit(batch_size))


# =========================
# Expand mapPath
# =========================
def expand_map_paths(data):
    rows = []

    for doc in data:
        raw_id = str(doc.get("_id", ""))
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
        if not isinstance(path, list):
            path = []

        if path:
            for i, pt in enumerate(path):
                if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                    continue
                lat, lon = pt
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

    df["ts_bucket"] = df["timestamp"].dt.floor(TIME_BUCKET)

    return df


# =========================
# Motion features
# =========================
def mph_to_mps(s):
    return pd.to_numeric(s, errors="coerce") * 0.44704


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

    # ⭐ REMOVED: Don't filter by MIN_POINTS for livestream - allow single points
    # counts = df.groupby("object_id")["timestamp"].transform("count")
    # df = df[counts >= MIN_POINTS_PER_OBJECT]

    if df.empty:
        return df

    df["dt_s"] = df.groupby("object_id")["timestamp"].diff().dt.total_seconds()
    df["dt_s"] = df["dt_s"].replace(0, np.nan)

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

    # ⭐ FIXED: Start with reported speed, then override with calculated where available
    df["speed_mps"] = mph_to_mps(df["speed"])  # Use reported speed as baseline

    df["speed_mps"] = np.where(
        (dist_m > 0.5) & (df["dt_s"] > 0),
        dist_m / df["dt_s"],
        df["speed_mps"]  # Keep reported speed if we can't calculate
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
    )

    df["zone_change"] = (
        df.groupby("object_id")["zones"].shift(1).astype(str) != df["zones"].astype(str)
    ).astype(int)

    df["path_gap"] = (
        df.groupby("object_id")["path_index"].diff().fillna(1) > 1
    ).astype(int)

    df["is_confident"] = df["certainty"] > CONFIDENCE_THRESHOLD

    accel_blocks = []
    for _, g in df.groupby("object_id"):
        if len(g) < 2:
            # Need at least 2 points for derivatives
            g["accel"] = np.nan
            g["jerk"] = np.nan
            accel_blocks.append(g)
            continue
        
        g = g.sort_values("timestamp").reset_index(drop=True)
        
        # Calculate derivatives directly on the original timestamps
        # No resampling - work with actual observation intervals
        speeds = g["speed_mps"].values
        times = g["timestamp"].values
        
        # Time deltas in seconds
        dt = np.diff(g["timestamp"].astype('int64') / 1e9)  # Convert to seconds
        dt = np.concatenate([[np.nan], dt])  # First point has no delta
        
        # Acceleration = change in speed / time
        dspeed = np.diff(speeds)
        accel = np.concatenate([[np.nan], dspeed / dt[1:]])
        
        # Smooth with rolling mean (min 2 points)
        accel_series = pd.Series(accel)
        accel_smooth = accel_series.rolling(window=3, min_periods=1, center=True).mean()
        
        # Jerk = change in acceleration / time  
        daccel = np.diff(accel_smooth)
        jerk = np.concatenate([[np.nan], daccel / dt[1:]])
        jerk_series = pd.Series(jerk)
        jerk_smooth = jerk_series.rolling(window=3, min_periods=1, center=True).mean()
        
        g["accel"] = accel_smooth.values
        g["jerk"] = jerk_smooth.values
        
        accel_blocks.append(g)

    return pd.concat(accel_blocks, ignore_index=True) if accel_blocks else pd.DataFrame()


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

    ttc = np.full_like(dist_xy, np.inf)
    valid = closing_rate < 0
    ttc[valid] = -dist_xy[valid] / closing_rate[valid]

    return g_pos, dist, rel_speed, heading_diff, closing_rate, ttc


def add_interaction_features(df):
    # ---- EARLY EXIT: not enough data ----
    if df.empty or "speed_mps" not in df.columns:
        for c in [
            "nn_dist_m",
            "rel_speed_mps",
            "heading_diff_deg",
            "closing_rate_mps",
            "ttc_s",
        ]:
            df[c] = np.nan
        return df

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

    for (_, _), g in df.groupby(["location", "ts_bucket"], sort=False):
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
# API ingestion
# =========================
def process_raw_docs(payload: list[dict]):
    if not payload:
        return pd.DataFrame()

    df = expand_map_paths(payload)
    df = add_motion_features(df)
    df = add_interaction_features(df)
    return df


# =========================
# Save + delete
# =========================
def save_stats(df):
    coll = get_stats_collection()

    if df.empty:
        return

    # ⭐ REMOVED: Don't filter by MIN_POINTS for livestream - save all observations
    # df = df[df.groupby("object_id")["object_id"].transform("count") >= MIN_POINTS_PER_OBJECT]

    keep = [
        "object_id", "timestamp", "location", "detected_type",
        "speed_mps", "accel", "jerk", "heading_deg", "d_heading_deg",
        "nn_dist_m", "closing_rate_mps", "ttc_s", "rel_speed_mps",
        "heading_diff_deg", "zone_change", "path_gap",
        "certainty", "is_confident", "lat", "lon"
    ]

    records = df[keep].dropna(subset=["timestamp", "object_id"]).to_dict("records")
    if records:
        coll.insert_many(records, ordered=False)


def delete_raw(raw_docs, chunk_size=5000):
    coll = get_raw_collection()
    ids = [d["_id"] for d in raw_docs]

    for i in range(0, len(ids), chunk_size):
        coll.delete_many({"_id": {"$in": ids[i:i + chunk_size]}})

    print(f"🧹 Deleted {len(ids)} raw docs")


# =========================
# Batch runner (LEGACY)
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
        
        # ⭐ For batch processing, filter to complete trajectories before saving
        df = df[df.groupby("object_id")["object_id"].transform("count") >= MIN_POINTS_PER_OBJECT]
        
        save_stats(df)
        delete_raw(raw)

        if len(raw) < batch_size:
            break


def load_all_combined_stats(limit: int = 10_000, location: str | None = None):
    config = configparser.ConfigParser()
    config.read("connection.ini")

    client = pymongo.MongoClient(config["DEFAULT"]["database"])
    db = client["camera-counts"]

    query = {}
    if location is not None:
        query["location"] = location

    cursor = (
        db["combined_stats"]
        .find(query)
        .sort("timestamp", -1)
        .limit(limit)
    )

    data = list(cursor)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).drop(columns=["_id"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# =========================
# CLI
# =========================
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=100_000)
    args = p.parse_args()
    run_batches(args.batch_size)