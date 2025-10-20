#!/usr/bin/env python3
"""
combined_statistics.py
Aggregates key vehicle statistics from MongoDB using the same configuration
as the main metadata pipeline, now extended to include (x, y) coordinates
for future near-accident analysis.
"""

import pymongo
import configparser
import pandas as pd
import numpy as np  # added for optional placeholder coordinates

# -------------------------------------------------------
# 1. Load same MongoDB connection as mongointerface.py
# -------------------------------------------------------
config = configparser.ConfigParser()
config.read("connection.ini")
dbUrl = config["DEFAULT"]["database"]

client = pymongo.MongoClient(dbUrl)
db = client["camera-counts"]  # same as in mongointerface.py
vehicleCollection = db["vehicles"]

# -------------------------------------------------------
# 2. Fetch all vehicle documents
# -------------------------------------------------------
def fetch_vehicle_data():
    data = list(vehicleCollection.find({}))
    print(f"✅ Retrieved {len(data)} vehicle records.")
    return data

# -------------------------------------------------------
# 3. Convert to DataFrame (added x/y logic)
# -------------------------------------------------------
def process_vehicle_data(data):
    rows = []
    for doc in data:
        row = {
            "timestamp": doc.get("timestamp"),
            "location": doc.get("location"),
            "detected_type": doc.get("detected_type"),
            "speed": doc.get("speed"),
            "certainty": doc.get("detection_certainty"),
            "zones": doc.get("zones", []),
            "time_elapsed": doc.get("time_elapsed"),
        }

        # Include lat/lon if available
        if "lat" in doc:
            row["lat"] = doc["lat"]
        else:
            row["lat"] = np.nan

        if "lon" in doc:
            row["lon"] = doc["lon"]
        else:
            row["lon"] = np.nan

        # -------------------------------------------------------
        # 🔵 Add derived X/Y coordinates for future analytics
        # -------------------------------------------------------
        # If lat/lon exist, temporarily use them as X/Y placeholders.
        # Later, when real Cartesian coordinates are added to the DB,
        # this section will automatically populate true positions.
        if not pd.isna(row["lat"]) and not pd.isna(row["lon"]):
            row["x"] = row["lon"]
            row["y"] = row["lat"]
        else:
            row["x"] = np.nan
            row["y"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)

# -------------------------------------------------------
# 4. Compute derived metrics
# -------------------------------------------------------
def compute_metrics(df):
    df["speed_mph"] = df["speed"]  # already converted in ffmpegreader
    df["is_confident"] = df["certainty"].fillna(0) > 0.5

    # 🔵 Optional: generate placeholder XY if DB has none
    if df["x"].isna().all() or df["y"].isna().all():
        np.random.seed(42)
        df["x"] = np.random.uniform(0, 100, len(df))
        df["y"] = np.random.uniform(0, 100, len(df))
        print("⚠️ No coordinates found in DB — generated random (x, y) placeholders.")

    return df

# -------------------------------------------------------
# 5. Run aggregation
# -------------------------------------------------------
if __name__ == "__main__":
    data = fetch_vehicle_data()
    df = process_vehicle_data(data)
    df = compute_metrics(df)

    print("\n✅ Combined Statistics Preview:")
    print(df.head())

    df.to_csv("combined_vehicle_stats.csv", index=False)
    print("\n💾 Saved to combined_vehicle_stats.csv")
