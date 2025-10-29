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

import argparse, configparser, math, numpy as np, pandas as pd, pymongo, concurrent.futures
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
        "_id": 1, "timestamp": 1, "location": 1, "detected_type": 1,
        "speed": 1, "detection_certainty": 1, "zones": 1,
        "time_elapsed": 1, "mapPath": 1
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
        base = {
            "object_id": str(doc.get("_id")),
            "timestamp": doc.get("timestamp"),
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
def mph_to_mps(s): return pd.to_numeric(s, errors="coerce").fillna(0) * 0.44704
def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360.0) % 360.0
def shortest_angle_diff_deg(a,b): return ((a - b + 180.0) % 360.0) - 180.0

# =========================
# Motion Features
# =========================
def add_motion_features(df):
    df = df.copy()
    df.sort_values(["object_id", "timestamp", "path_index"], inplace=True)
    df["speed_mps"] = mph_to_mps(df["speed"])
    df["is_confident"] = df["certainty"].fillna(0) > CONFIDENCE_THRESHOLD

    dt = df.groupby("object_id")["time_elapsed"].diff().fillna(0.1)
    dv = df.groupby("object_id")["speed_mps"].diff().fillna(0.0)
    df["accel"] = (dv / dt.replace(0, np.nan)).fillna(0.0).astype("float32")
    df["jerk"] = (df.groupby("object_id")["accel"].diff() / dt.replace(0, np.nan)).fillna(0.0).astype("float32")

    lat_prev = df.groupby("object_id")["lat"].shift(1)
    lon_prev = df.groupby("object_id")["lon"].shift(1)
    df["heading_deg"] = bearing_deg(lat_prev, lon_prev, df["lat"], df["lon"]).fillna(0.0).astype("float32")
    prev_heading = df.groupby("object_id")["heading_deg"].shift(1)
    df["d_heading_deg"] = shortest_angle_diff_deg(df["heading_deg"], prev_heading).fillna(0.0).astype("float32")

    prev_z = df.groupby("object_id")["zones"].shift(1)
    df["zone_change"] = (prev_z.astype(str) != df["zones"].astype(str)).astype("int8")
    df["path_gap"] = (df.groupby("object_id")["path_index"].diff().fillna(1) > 1).astype("int8")
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
    vx, vy = spd*np.cos(np.radians(hdg)), spd*np.sin(np.radians(hdg))

    pos_of_label = pd.Series(np.arange(n), index=df.index)
    group_iter = df.groupby(["location", "timestamp"], sort=False)

    tasks = []
    for (loc, t), g in group_iter:
        if len(g) <= 1: continue
        if np.random.randint(0, sample_step) != 0: continue
        g_pos = pos_of_label.loc[g.index].to_numpy()
        tasks.append((loc, t, g_pos, lat, lon, spd, hdg, vx, vy))

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for result in tqdm(ex.map(compute_interaction_for_group, tasks), total=len(tasks), desc="Interactions"):
            if result is None: continue
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
        "object_id","timestamp","location","detected_type",
        "speed_mps","accel","jerk","heading_deg","d_heading_deg",
        "nn_dist_m","closing_rate_mps","ttc_s","rel_speed_mps","heading_diff_deg",
        "zone_change","path_gap","certainty","is_confident","lat","lon"
    ]
    df[keep].to_csv("combined_vehicle_stats_expanded.csv", index=False)
    print("💾 Saved → combined_vehicle_stats_expanded.csv")

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
        for c in ["nn_dist_m","rel_speed_mps","heading_diff_deg","closing_rate_mps","ttc_s"]:
            df[c] = np.nan

    save_results(df)
