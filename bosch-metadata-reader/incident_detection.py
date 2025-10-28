#!/usr/bin/env python3
"""
incident_detection.py
FINAL version — supports retraining and reusing existing models.

Usage:
    python3 incident_detection.py train     # retrains + saves new model
    python3 incident_detection.py existing  # loads latest saved model
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os, sys, pickle
from datetime import datetime

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
# 3. Train + save or load model
# -------------------------------------------------------
def train_global_model(df):
    X = df[["speed", "lat_scaled", "lon_scaled"]].fillna(0).values
    X[:, 0] *= 3.0  # emphasize speed

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    print("\n🧠 Training global IsolationForest (learning normal traffic pattern)...")
    model = IsolationForest(
        n_estimators=400,
        contamination=0.0000,
        random_state=42,
        n_jobs=-1
    ).fit(X_scaled)
    print("✅ Model trained successfully.")

    # Save both model + scaler
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    model_path = f"models/model_{timestamp}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    print(f"💾 Saved model and scaler → {model_path}")
    return model, scaler


def load_latest_model():
    os.makedirs("models", exist_ok=True)
    model_files = [f for f in os.listdir("models") if f.endswith(".pkl")]
    if not model_files:
        raise FileNotFoundError("❌ No saved models found in 'models/' — please train first.")

    latest = max(model_files, key=lambda x: os.path.getmtime(os.path.join("models", x)))
    latest_path = os.path.join("models", latest)

    with open(latest_path, "rb") as f:
        data = pickle.load(f)

    print(f"📂 Loaded existing model from {latest_path}")
    return data["model"], data["scaler"]


# -------------------------------------------------------
# 4. Frame-by-frame detection
# -------------------------------------------------------
def detect_and_visualize(df, model, scaler, bounds):
    timestamps = sorted(df["timestamp"].dropna().unique())
    print(f"\n🕒 Processing {len(timestamps)} frames...\n")

    plt.ion()
    total_anomalies = 0

    try:
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

            window_size = 5
            rolling_flags = []

            for t in timestamps:
                if not plt.fignum_exists(fig.number):
                    raise KeyboardInterrupt

                frame = subset[subset["timestamp"] == t].copy()
                if len(frame) < 5:
                    continue

                X = frame[["speed", "lat_scaled", "lon_scaled"]].fillna(0).values
                X[:, 0] *= 3.0
                X_scaled = scaler.transform(X)

                frame["is_anomaly"] = model.predict(X_scaled)
                frame["anomaly_score"] = model.decision_function(X_scaled)
                num_anom = (frame["is_anomaly"] == -1).sum()
                total_anomalies += num_anom

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

            plt.close(fig)

    except KeyboardInterrupt:
        print("\n🛑 Visualization interrupted by user (window closed).")

    finally:
        plt.ioff()
        print(f"\n📊 TOTAL anomalies detected across all frames: {total_anomalies}")


# -------------------------------------------------------
# 5. Main
# -------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "existing"):
        print("Usage: python3 incident_detection.py [train|existing]")
        sys.exit(1)

    mode = sys.argv[1]
    df = load_data()
    df, bounds = scale_per_location(df)

    if mode == "train":
        model, scaler = train_global_model(df)
    else:
        model, scaler = load_latest_model()

    detect_and_visualize(df, model, scaler, bounds)
