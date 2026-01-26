#!/usr/bin/env python3
import pickle
import numpy as np
import pandas as pd

CSV_PATH = "combined_vehicle_stats_expandedNEW.csv"
PKL_PATH = "models_by_loc_2026-01-09_10-59.pkl"
LOC_KEY  = "foothill"

# Pick a small, safe starter set of numeric features from your CSV.
# Adjust if your column names differ.
FEATURES = [
    "speed_mps",
    "accel",
    "jerk",
    "heading_deg",
    "d_heading_deg",
    "nn_dist_m",
    "closing_rate_mps",
    "ttc_s",
    "rel_speed_mps",
    "heading_diff_deg",
    "zone_change",
    "path_gap",
    "certainty",
]

def main():
    print("reading csv...")
    df = pd.read_csv(CSV_PATH)
    print("rows:", len(df))
    print("cols:", len(df.columns))

    # Filter to 1 location so we match the model key
    if "location" in df.columns:
        df = df[df["location"] == LOC_KEY].copy()
        print("rows after loc filt:", len(df))

    # Make sure features exist
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing feature cols: {missing}")

    # Build X and clean inf/NaN
    X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    X = X.to_numpy(dtype=float)

    # Just test a small slice first
    Xs = X[:200]

    print("loading pkl...")
    with open(PKL_PATH, "rb") as f:
        models = pickle.load(f)

    print("top keys:", list(models.keys())[:10])
    model = models[LOC_KEY]["model"]
    print("model type:", type(model))

    print("running predict...")
    preds = model.predict(Xs)                 # -1 anomaly, 1 normal
    scores = model.decision_function(Xs)      # lower = more abnormal

    n_anom = int((preds == -1).sum())
    print("sample anomalies:", n_anom, "/", len(preds))
    print("score min/mean/max:", float(scores.min()), float(scores.mean()), float(scores.max()))

if __name__ == "__main__":
    main()
