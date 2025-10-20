#!/usr/bin/env python3
"""
incident_detection.py
Performs unsupervised anomaly detection on combined vehicle statistics.
Uses IsolationForest to flag potentially abnormal movements or near-incidents.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os

# -------------------------------------------------------
# 1. Load data
# -------------------------------------------------------
def load_data(csv_path="combined_vehicle_stats.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} records from {csv_path}")
    return df

# -------------------------------------------------------
# 2. Prepare features for the model
# -------------------------------------------------------
def prepare_features(df):
    # choose numeric features likely to reflect anomalies
    features = ["speed", "x", "y", "time_elapsed"]
    # handle missing data
    df[features] = df[features].fillna(0)
    X = df[features].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, features

# -------------------------------------------------------
# 3. Run IsolationForest
# -------------------------------------------------------
def detect_anomalies(df, X_scaled, contamination=0.05):
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        bootstrap=True,
        n_jobs=-1
    )
    # 🔧 fit the model first!
    model.fit(X_scaled)

    # then compute anomaly scores + labels
    df["anomaly_score"] = model.decision_function(X_scaled)
    df["is_anomaly"] = model.predict(X_scaled)  # -1 = anomaly, 1 = normal
    return df, model

# -------------------------------------------------------
# 4. Visualize anomalies (optional)
# -------------------------------------------------------
def visualize(df):
    plt.figure(figsize=(8,6))
    colors = np.where(df["is_anomaly"] == -1, "red", "blue")
    plt.scatter(df["x"], df["y"], c=colors, alpha=0.6, s=30)
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Detected Vehicle Anomalies (red = anomaly)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------------------------------------------------
# 5. Save results
# -------------------------------------------------------
def save_results(df, out_csv="vehicle_anomaly_scores.csv"):
    df.to_csv(out_csv, index=False)
    print(f"💾 Saved anomaly results → {out_csv}")
    print(f"🚨 {sum(df['is_anomaly']==-1)} anomalies detected")

# -------------------------------------------------------
# 6. Main pipeline
# -------------------------------------------------------
if __name__ == "__main__":
    df = load_data()
    X_scaled, features = prepare_features(df)
    df, model = detect_anomalies(df, X_scaled, contamination=0.05)
    save_results(df)
    visualize(df)
