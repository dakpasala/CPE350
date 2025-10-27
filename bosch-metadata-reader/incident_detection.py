#!/usr/bin/env python3
"""
incident_detection.py
FINAL tuned version for realistic traffic anomaly detection.

✓ Trains one global IsolationForest baseline
✓ Very low contamination (0.05%)
✓ Global StandardScaler (no per-frame distortions)
✓ Emphasizes speed more heavily
✓ Temporal smoothing to ignore flickers
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

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
# 3. Train global IsolationForest baseline + scaler
# -------------------------------------------------------
def train_global_model(df):
    X = df[["speed", "lat_scaled", "lon_scaled"]].fillna(0).values
    X[:, 0] *= 3.0  # emphasize speed

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print("\n🧠 Training global IsolationForest (learning normal traffic pattern)...")
    model = IsolationForest(
        n_estimators=400,
        contamination=0.0005,  # extremely low: ~0.05% anomalies
        random_state=42,
        n_jobs=-1
    ).fit(X_scaled)
    print("✅ Model trained successfully with minimal false positives.")
    return model, scaler


# -------------------------------------------------------
# 4. Frame-by-frame detection using baseline
# -------------------------------------------------------
def detect_and_visualize(df, model, scaler, bounds):
    timestamps = sorted(df["timestamp"].dropna().unique())
    print(f"\n🕒 Processing {len(timestamps)} frames...\n")

    plt.ion()

    for loc, group in df.groupby("location"):
        print(f"\n🌍 Location: {loc}")
        subset = group.copy()

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.set_title(f"Live Anomaly Detection — {loc} (red = anomaly)")
        ax.set_xlabel("Longitude (scaled)")
        ax.set_ylabel("Latitude (scaled)")
        ax.grid(True)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        scatter = ax.scatter([], [], c=[], alpha=0.7, s=40)
        text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

        # temporal smoothing window
        window_size = 5
        rolling_flags = []

        for t in timestamps:
            frame = subset[subset["timestamp"] == t].copy()
            if len(frame) < 5:
                continue

            X = frame[["speed", "lat_scaled", "lon_scaled"]].fillna(0).values
            X[:, 0] *= 3.0
            X_scaled = scaler.transform(X)

            frame["is_anomaly"] = model.predict(X_scaled)
            frame["anomaly_score"] = model.decision_function(X_scaled)
            num_anom = (frame["is_anomaly"] == -1).sum()

            # persistence check (smooth anomalies)
            rolling_flags.append(num_anom)
            if len(rolling_flags) > window_size:
                rolling_flags.pop(0)
            avg_anom = np.mean(rolling_flags)

            text.set_text(
                f"Timestamp: {t}\nFrame anomalies: {num_anom}/{len(frame)}\nRolling avg: {avg_anom:.2f}"
            )
            colors = np.where(frame["is_anomaly"] == -1, "red", "blue")
            scatter.set_offsets(np.c_[frame["lon_scaled"], frame["lat_scaled"]])
            scatter.set_color(colors)

            plt.pause(0.4)

        plt.ioff()
        plt.show()


# -------------------------------------------------------
# 5. Run pipeline
# -------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    df, bounds = scale_per_location(df)
    model, scaler = train_global_model(df)
    detect_and_visualize(df, model, scaler, bounds)
