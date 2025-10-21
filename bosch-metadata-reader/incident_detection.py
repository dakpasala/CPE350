#!/usr/bin/env python3
"""
incident_detection.py
Frame-by-frame anomaly detection with persistent live plot.
Now scales per location (Foothill, Dunbarton, etc.) so data doesn't flatten,
and keeps fixed axis ranges for each location.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import os
import time

# -------------------------------------------------------
# 1. Load expanded dataset
# -------------------------------------------------------
def load_data(csv_path="combined_vehicle_stats_expanded.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")
    print(f"✅ Loaded {len(df)} records from {csv_path}")
    return df

# -------------------------------------------------------
# 2. Compute per-location scaling (0–1)
# -------------------------------------------------------
def scale_per_location(df):
    scaled = []
    bounds = {}
    for loc, group in df.groupby("location"):
        lat_min, lat_max = group["lat"].min(), group["lat"].max()
        lon_min, lon_max = group["lon"].min(), group["lon"].max()

        df.loc[group.index, "lat_scaled"] = (group["lat"] - lat_min) / (lat_max - lat_min)
        df.loc[group.index, "lon_scaled"] = (group["lon"] - lon_min) / (lon_max - lon_min)

        bounds[loc] = (lon_min, lon_max, lat_min, lat_max)
        print(f"📍 {loc}: lat({lat_min:.6f}–{lat_max:.6f}), lon({lon_min:.6f}–{lon_max:.6f})")
    return df, bounds

# -------------------------------------------------------
# 3. Frame-by-frame anomaly detection and live visualization
# -------------------------------------------------------
def detect_and_visualize(df, bounds, contamination=0.03):
    timestamps = sorted(df["timestamp"].dropna().unique())
    print(f"\n🕒 Processing {len(timestamps)} frames...\n")

    for loc, group in df.groupby("location"):
        print(f"\n🌍 Location: {loc}")
        loc_bounds = bounds[loc]
        subset = group.copy()

        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title(f"Live Anomaly Detection — {loc} (red = anomaly)")
        ax.set_xlabel("Longitude (scaled)")
        ax.set_ylabel("Latitude (scaled)")
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        scatter = ax.scatter([], [], c=[], alpha=0.7, s=40)
        text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

        for t in timestamps:
            frame = subset[subset["timestamp"] == t].copy()
            if len(frame) < 5:
                continue

            X = frame[["speed", "lat_scaled", "lon_scaled"]].fillna(0).values
            X = StandardScaler().fit_transform(X)

            model = IsolationForest(
                n_estimators=200,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            ).fit(X)

            frame["anomaly_score"] = model.decision_function(X)
            frame["is_anomaly"] = model.predict(X)
            num_anom = (frame["is_anomaly"] == -1).sum()

            # Update live plot
            text.set_text(f"Timestamp: {t}\nAnomalies: {num_anom}/{len(frame)}")
            colors = np.where(frame["is_anomaly"] == -1, "red", "blue")
            scatter.set_offsets(np.c_[frame["lon_scaled"], frame["lat_scaled"]])
            scatter.set_color(colors)

            plt.pause(0.5)

        plt.ioff()
        plt.show()

# -------------------------------------------------------
# 4. Run pipeline
# -------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    df, bounds = scale_per_location(df)
    detect_and_visualize(df, bounds, contamination=0.03)
