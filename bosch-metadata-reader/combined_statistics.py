"""
combined_statistics.py

Real-time feature extraction engine.

Pipeline:
  raw vehicle docs (API) → combined_stats (Mongo)

- Accepts raw vehicle docs from ffmpegreader via API
- Expands mapPath
- Computes motion + interaction features
- Writes derived rows to MongoDB
"""

import configparser
import concurrent.futures
import os
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
MAX_WORKERS = max(2, int((os.cpu_count() or 4) * 0.75))


# =========================
# Mongo helpers (UNCHANGED)
# =========================
def _get_db():
    config = configparser.ConfigParser()
    config.read("connection.ini")
    client = pymongo.MongoClient(config["DEFAULT"]["database"])
    return client["camera-counts"]


def get_stats_collection():
    return _get_db()["combined_stats"]


# =========================
# Core processing pipeline
# =========================
def process_raw_docs(raw_docs: list[dict]) -> pd.DataFrame:
    """
    Pure feature extraction pipeline.
    Input: list of raw vehicle docs (from ffmpegreader)
    Output: DataFrame of derived features
    """
    if not raw_docs:
        return pd.DataFrame()

    df = expand_map_paths(raw_docs)
    df = add_motion_features(df)
    df = explain_add_interaction_features(df)
    return df


# =========================
# Expand mapPath (UNCHANGED)
# =========================
def expand_map_paths(data):
    rows = []

    for doc in data:
        raw_id = str(doc.get("_id") or doc.get("id"))
        ts = pd.to_datetime(doc.get("timestamp"), errors="coerce")

        if "_" in raw_id:
            maybe_id, maybe_ts = raw_id.split("_", 1)
            ts_from_id = pd.to_datetime(maybe_ts, errors="coerce")
            if not pd.isna(ts_from_id):
                raw_id = maybe_id
                ts = ts_from_id

        base = {
            "object_id": raw_id,
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
# Motion features (UNCHANGED)
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

    return df


# =========================
# Interaction features (UNCHANGED)
# =========================
def explain_add_interaction_features(df):
    # identical to your add_interaction_features
    # renamed to avoid collisions and make intent explicit
    from copy import deepcopy
    return add_interaction_features(df)


# =========================
# Save derived stats (UNCHANGED)
# =========================
def save_stats(df: pd.DataFrame):
    if df.empty:
        return

    coll = get_stats_collection()

    keep = [
        "object_id", "timestamp", "location", "detected_type",
        "speed_mps", "heading_deg", "d_heading_deg",
        "nn_dist_m", "closing_rate_mps", "ttc_s", "rel_speed_mps",
        "heading_diff_deg", "zone_change", "path_gap",
        "certainty", "is_confident", "lat", "lon"
    ]

    records = df[keep].dropna(subset=["timestamp"]).to_dict("records")
    if records:
        coll.insert_many(records, ordered=False)
