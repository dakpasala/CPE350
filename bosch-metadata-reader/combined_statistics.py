#!/usr/bin/env python3
"""
combined_statistics.py
Expands each vehicle's mapPath points into individual rows.
Each mapPath[i] entry (lat, lon) becomes its own row with same vehicle metadata.
Output is sorted by timestamp ascending for time-series analysis.
"""

import pymongo
import configparser
import pandas as pd
import numpy as np
from bson import ObjectId

# -------------------------------------------------------
# 1. Load MongoDB connection
# -------------------------------------------------------
config = configparser.ConfigParser()
config.read("connection.ini")
dbUrl = config["DEFAULT"]["database"]

client = pymongo.MongoClient(dbUrl)
db = client["camera-counts"]
vehicleCollection = db["vehicles"]

# -------------------------------------------------------
# 2. Fetch all vehicle documents
# -------------------------------------------------------
def fetch_vehicle_data():
    data = list(vehicleCollection.find({}))
    print(f"✅ Retrieved {len(data)} vehicle records.")
    return data

# -------------------------------------------------------
# 3. Expand mapPath into multiple rows per vehicle
# -------------------------------------------------------
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

        map_path = doc.get("mapPath", [])
        if isinstance(map_path, list) and len(map_path) > 0:
            for i, coords in enumerate(map_path):
                if isinstance(coords, (list, tuple)) and len(coords) == 2:
                    lat, lon = coords
                    row = base.copy()
                    row["path_index"] = i
                    row["lat"] = lat
                    row["lon"] = lon
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

    return pd.DataFrame(rows)

# -------------------------------------------------------
# 4. Compute derived metrics
# -------------------------------------------------------
def compute_metrics(df):
    df["speed_mph"] = df["speed"]
    df["is_confident"] = df["certainty"].fillna(0) > 0.5
    return df

# -------------------------------------------------------
# 5. Sort by timestamp ascending
# -------------------------------------------------------
def sort_by_time(df):
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    return df

# -------------------------------------------------------
# 6. Run aggregation
# -------------------------------------------------------
if __name__ == "__main__":
    data = fetch_vehicle_data()
    df = expand_map_paths(data)
    df = compute_metrics(df)
    df = sort_by_time(df)

    print("\n✅ Expanded and sorted dataset preview:")
    print(df.head())

    df.to_csv("combined_vehicle_stats_expanded.csv", index=False)
    print("\n💾 Saved expanded data → combined_vehicle_stats_expanded.csv")
