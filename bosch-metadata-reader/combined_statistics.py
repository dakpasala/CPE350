#!/usr/bin/env python3
"""
combined_statistics.py
Expands each vehicle's mapPath points into individual rows, then computes:
  • Motion features (per vehicle): speed_mps, accel, jerk, heading, d_heading, zone_change, path_gap
  • Interaction features (per (location, timestamp)): nn_dist_m, rel_speed_mps, heading_diff_deg, closing_rate_mps, ttc_s

This version is MEMORY-SAFE:
  - No giant lists of DataFrames; writes interaction features in-place.
  - Downcasts numeric dtypes where safe.
  - Optional flags for quick/low-RAM runs.

Usage:
  python3 combined_statistics.py [--no-interactions] [--limit N]
"""

import argparse
import configparser
import math
import numpy as np
import pandas as pd
import pymongo
from sklearn.neighbors import BallTree

# =========================
# Config / constants
# =========================
R_EARTH_M = 6371000.0
CONFIDENCE_THRESHOLD = 0.5

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
    cursor = coll.find({}, projection={
        "_id": 1,
        "timestamp": 1,
        "location": 1,
        "detected_type": 1,
        "speed": 1,
        "detection_certainty": 1,
        "zones": 1,
        "time_elapsed": 1,
        "mapPath": 1,
    })
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
        base = {
            "object_id": str(doc.get("_id")),
            "timestamp": doc.get("timestamp"),
            "location": doc.get("location"),
            "detected_type": doc.get("detected_type"),
            "speed": doc.get("speed"),
            "certainty": doc.get("detection_certainty"),
            "zones": doc.get("zones", []),
            "time_elapsed": doc.get("time_elapsed"),
        }
        path = doc.get("mapPath", [])
        if isinstance(path, list) and path:
            for i, coords in enumerate(path):
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    lat, lon = coords
                    row = base.copy()
                    row["path_index"] = i
                    row["lat"] = lat
                    row["lon"] = lon
                    # keep x/y for compatibility (lon/lat)
                    row["x"] = lon
                    row["y"] = lat
                    rows.append(row)
        else:
            row = base.copy()
            row["path_index"] = np.nan
            row["lat"] = np.nan
            row["lon"] = np.nan
            row["x"] = np.nan
            row["y"] = np.nan
            rows.append(row)

    df = pd.DataFrame(rows)
    # Downcast early to save memory
    for col in ["lat", "lon", "x", "y", "speed"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    if "path_index" in df:
        df["path_index"] = pd.to_numeric(df["path_index"], errors="coerce").astype("float32")
    return df

# =========================
# Basic derived metrics
# =========================
def compute_metrics(df):
    df["speed_mph"] = df["speed"].astype("float32")
    df["is_confident"] = df["certainty"].fillna(0) > CONFIDENCE_THRESHOLD
    return df

# =========================
# Sorting
# =========================
def sort_by_time(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values(by=["timestamp"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# =========================
# Kinematics helpers
# =========================
def mph_to_mps(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype("float32") * 0.44704

def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    brng = (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
    return brng

def shortest_angle_diff_deg(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return d

# =========================
# Motion features (per vehicle)
# =========================
def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.sort_values(["object_id", "timestamp"], inplace=True)

    df["speed_mps"] = mph_to_mps(df["speed"])

    dt = df.groupby("object_id")["timestamp"].diff().dt.total_seconds().astype("float32")
    dv = df.groupby("object_id")["speed_mps"].diff().astype("float32")

    df["accel"] = (dv / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    df["jerk"]  = (df.groupby("object_id")["accel"].diff() / dt).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")

    lat_prev = df.groupby("object_id")["lat"].shift(1)
    lon_prev = df.groupby("object_id")["lon"].shift(1)
    df["heading_deg"] = bearing_deg(lat_prev, lon_prev, df["lat"], df["lon"])
    df["heading_deg"] = df.groupby("object_id")["heading_deg"].ffill().fillna(0.0).astype("float32")

    prev_heading = df.groupby("object_id")["heading_deg"].shift(1)
    df["d_heading_deg"] = shortest_angle_diff_deg(df["heading_deg"], prev_heading).fillna(0.0).astype("float32")

    # zones may be lists; compare string form
    prev_z = df.groupby("object_id")["zones"].shift(1)
    df["zone_change"] = (prev_z.astype(str) != df["zones"].astype(str)).astype("int8")

    df["path_gap"] = (df.groupby("object_id")["path_index"].diff().fillna(1) > 1).astype("int8")

    return df

# =========================
# Interaction features (memory-safe)
# =========================
def add_interaction_features_inplace(df: pd.DataFrame) -> pd.DataFrame:
    """
    Writes interaction features directly into df (preallocated columns),
    grouped by (location, timestamp). Uses positional indexing robustly.
    """
    df = df.copy()

    # Preallocate outputs
    n = len(df)
    df["nn_dist_m"]        = np.full(n, np.nan, dtype="float32")
    df["rel_speed_mps"]    = np.full(n, np.nan, dtype="float32")
    df["heading_diff_deg"] = np.full(n, np.nan, dtype="float32")
    df["closing_rate_mps"] = np.full(n, np.nan, dtype="float32")
    df["ttc_s"]            = np.full(n, np.nan, dtype="float32")

    # Build arrays aligned to current df row order (positional)
    # (Use to_numpy() so we can index by positional integer arrays)
    lat = pd.to_numeric(df["lat"], errors="coerce").astype("float64").to_numpy()
    lon = pd.to_numeric(df["lon"], errors="coerce").astype("float64").to_numpy()
    spd = pd.to_numeric(df["speed_mps"], errors="coerce").fillna(0.0).astype("float32").to_numpy()
    hdg = pd.to_numeric(df["heading_deg"], errors="coerce").fillna(0.0).astype("float32").to_numpy()

    theta = np.radians(hdg).astype("float64")
    vx = (spd * np.cos(theta)).astype("float32")
    vy = (spd * np.sin(theta)).astype("float32")

    # Fast label -> position mapping for this df
    # (So we can convert group index labels to positional indices)
    pos_of_label = pd.Series(np.arange(n, dtype=np.int64), index=df.index)

    # Group once (labels), then map to positions each loop
    for (loc, t), g in df.groupby(["location", "timestamp"], sort=False):
        # Label indices for this group
        g_labels = g.index.to_numpy()
        # Positional indices (0..n-1)
        g_pos = pos_of_label.loc[g_labels].to_numpy()

        if g_pos.size <= 1:
            continue

        # Build BallTree on this group's points (in radians)
        lat_rad = np.radians(lat[g_pos])
        lon_rad = np.radians(lon[g_pos])
        pts = np.column_stack([lat_rad, lon_rad])

        tree = BallTree(pts, metric="haversine")
        d_rad, nbr_idx = tree.query(pts, k=2)
        nn_local = nbr_idx[:, 1]              # local neighbor index within this group
        nn_dist = (d_rad[:, 1] * R_EARTH_M).astype("float32")

        # Relative speed
        sp = spd[g_pos]
        rel_speed = np.abs(sp - sp[nn_local]).astype("float32")

        # Heading difference (shortest)
        hd = hdg[g_pos]
        hd_nn = hd[nn_local]
        heading_diff = np.abs(((hd - hd_nn + 180.0) % 360.0) - 180.0).astype("float32")

        # LOS geometry in meters (local ENU approx)
        dlat = lat_rad - lat_rad[nn_local]
        dlon = lon_rad - lon_rad[nn_local]
        dx = (dlon * np.cos((lat_rad + lat_rad[nn_local]) / 2.0) * R_EARTH_M).astype("float32")
        dy = (dlat * R_EARTH_M).astype("float32")
        dist = np.hypot(dx, dy).astype("float32")

        # unit LOS vector
        with np.errstate(divide="ignore", invalid="ignore"):
            ux = np.divide(dx, dist, out=np.zeros_like(dist), where=dist > 0)
            uy = np.divide(dy, dist, out=np.zeros_like(dist), where=dist > 0)

        # Relative velocity projected onto LOS (closing rate)
        vx_i = vx[g_pos]
        vy_i = vy[g_pos]
        rvx = (vx_i - vx_i[nn_local]).astype("float32")
        rvy = (vy_i - vy_i[nn_local]).astype("float32")
        closing_rate = (rvx * ux + rvy * uy).astype("float32")

        with np.errstate(divide="ignore", invalid="ignore"):
            ttc = np.where(closing_rate < 0, -dist / closing_rate, np.inf).astype("float32")

        # Write back by positional indices via iloc
        df.iloc[g_pos, df.columns.get_loc("nn_dist_m")]        = nn_dist
        df.iloc[g_pos, df.columns.get_loc("rel_speed_mps")]    = rel_speed
        df.iloc[g_pos, df.columns.get_loc("heading_diff_deg")] = heading_diff
        df.iloc[g_pos, df.columns.get_loc("closing_rate_mps")] = closing_rate
        df.iloc[g_pos, df.columns.get_loc("ttc_s")]            = ttc

    return df

# =========================
# Main
# =========================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--no-interactions", action="store_true",
                   help="Skip nearest-neighbor interaction features (faster, lower RAM).")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit number of vehicle docs fetched from Mongo (for quick tests).")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    data = fetch_vehicle_data(limit=args.limit)
    df = expand_map_paths(data)
    df = compute_metrics(df)
    df = sort_by_time(df)

    # Motion features are cheap – always do them
    df = add_motion_features(df)

    # Interaction features can be heavy – optional
    if not args.no_interactions:
        # Optionally restrict to confident rows before interactions to reduce work
        # (comment out if you want interactions for all rows)
        df_conf = df[df["is_confident"].fillna(False)].copy()
        if len(df_conf) > 1:
            # compute in-place on confident subset…
            df_conf = add_interaction_features_inplace(df_conf)
            # …then merge back the columns (preserve NaNs for others)
            cols = ["object_id", "timestamp", "nn_dist_m", "rel_speed_mps",
                    "heading_diff_deg", "closing_rate_mps", "ttc_s"]
            df = df.merge(df_conf[cols], on=["object_id", "timestamp"], how="left")
        else:
            for c in ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]:
                df[c] = np.nan
    else:
        for c in ["nn_dist_m", "rel_speed_mps", "heading_diff_deg", "closing_rate_mps", "ttc_s"]:
            df[c] = np.nan

    # Final downcast where safe to shrink CSV size
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")

    print("\n✅ Expanded + enriched dataset preview:")
    print(df.head())

    df.to_csv("combined_vehicle_stats_expanded.csv", index=False)
    print("\n💾 Saved → combined_vehicle_stats_expanded.csv")
