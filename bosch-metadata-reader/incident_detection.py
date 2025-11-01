#!/usr/bin/env python3
"""
incident_detection.py
Adaptive per-location IsolationForest over motion + interaction features.
Trains only on "normal" data (your CSV) to learn baseline traffic flow.

Usage:
    python3 incident_detection.py train [csv_path]
    python3 incident_detection.py existing [csv_path]
"""

import os, sys, pickle
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# =============================================================================
# Config
# =============================================================================

DEFAULT_CSV = "combined_vehicle_stats_expanded.csv"
MODELS_DIR = "models"
CUTOFF_QUANTILE = 0.00    # bottom 1% considered anomalous
PERSIST_WINDOW = 3
PERSIST_MIN = 2
MIN_FRAME_POINTS = 5
PAUSE_SEC = 0.35

# corroboration thresholds
NEAR_DIST_M = 8.0
NEAR_HEADING_DIFF = 45.0
NEAR_TTC_S = 3.0

# =============================================================================
# Load CSV + scale lat/lon
# =============================================================================

def load_data(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")
    return df


def scale_per_location(df: pd.DataFrame):
    df = df.copy()
    bounds = {}
    for loc, g in df.groupby("location"):
        lat_min, lat_max = g["lat"].min(), g["lat"].max()
        lon_min, lon_max = g["lon"].min(), g["lon"].max()
        lat_rng = (lat_max - lat_min) if lat_max != lat_min else 1.0
        lon_rng = (lon_max - lon_min) if lon_max != lon_min else 1.0

        idx = g.index
        df.loc[idx, "lat_scaled"] = (g["lat"] - lat_min) / lat_rng
        df.loc[idx, "lon_scaled"] = (g["lon"] - lon_min) / lon_rng
        bounds[loc] = (lon_min, lon_max, lat_min, lat_max)
    print("📍 Scaled latitude/longitude per location.")
    return df, bounds


# =============================================================================
# Feature selection
# =============================================================================

def _present(df, cols):
    return [c for c in cols if c in df.columns]

def get_feature_columns(df):
    primary = [
        "speed_mps", "accel", "jerk",
        "d_heading_deg", "zone_change", "path_gap",
        "nn_dist_m", "rel_speed_mps", "heading_diff_deg",
        "closing_rate_mps", "ttc_s",
        "lat_scaled", "lon_scaled",
    ]
    feats = _present(df, primary)
    if not feats:
        print("⚠️ Enriched features missing; falling back to base columns.")
        feats = _present(df, ["speed", "lat_scaled", "lon_scaled"])
    return feats


# =============================================================================
# Drift + alignment helpers
# =============================================================================

def align_features(df, model_features):
    """Ensure df columns match model expectations."""
    aligned = df.copy()
    for f in model_features:
        if f not in aligned.columns:
            aligned[f] = 0.0
    aligned = aligned[model_features]
    return aligned

def detect_drift(scaler, new_X, threshold=0.25):
    old_mean, old_std = scaler.mean_, scaler.scale_
    new_mean, new_std = np.mean(new_X, axis=0), np.std(new_X, axis=0)
    mean_drift = np.mean(np.abs(new_mean - old_mean) / (old_std + 1e-8))
    std_drift = np.mean(np.abs(new_std - old_std) / (old_std + 1e-8))
    drift = max(mean_drift, std_drift)
    return drift > threshold, drift


# =============================================================================
# Train / Save / Load
# =============================================================================

def train_by_location(df: pd.DataFrame):
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = {}
    features = get_feature_columns(df)
    if not features:
        raise RuntimeError("No usable features to train on.")

    print("\n🧠 Training IsolationForest per location...")
    for loc, g in df.groupby("location"):
        g = g[g["is_confident"].fillna(True)] if "is_confident" in g else g
        X = g[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        if len(X) < 50:
            print(f"   • Skipping {loc}: only {len(X)} samples.")
            continue

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        model = IsolationForest(
            n_estimators=400,
            contamination=0.001,
            random_state=42,
            n_jobs=-1
        ).fit(Xs)

        scores = model.decision_function(Xs)
        cut = np.quantile(scores, CUTOFF_QUANTILE)
        models[loc] = {"model": model, "scaler": scaler, "features": features, "cut": float(cut)}
        print(f"   • {loc}: trained ({len(X)} pts), cutoff={cut:.5f}")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = os.path.join(MODELS_DIR, f"models_by_loc_{ts}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\n💾 Models saved → {out_path}")
    return models


def load_latest_models():
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError("❌ Models directory not found.")
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError("❌ No saved models in /models.")
    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)))
    path = os.path.join(MODELS_DIR, latest)
    with open(path, "rb") as f:
        models = pickle.load(f)
    print(f"📂 Loaded models from {path}")
    return models


def maybe_retrain_if_drifted(df, models):
    updated = 0
    for loc, g in df.groupby("location"):
        if loc not in models:
            print(f"🆕 New location {loc} → training new model.")
            new_model = train_by_location(g)
            if loc in new_model:
                models[loc] = new_model[loc]
                updated += 1
            continue

        entry = models[loc]
        feats = entry["features"]
        X = align_features(g, feats).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        Xs = entry["scaler"].transform(X)
        drifted, val = detect_drift(entry["scaler"], Xs)
        if drifted:
            print(f"⚠️ Drift detected for {loc} (drift={val:.2f}) → retraining...")
            retrained = train_by_location(g)
            if loc in retrained:
                models[loc] = retrained[loc]
                updated += 1

    if updated:
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
        path = os.path.join(MODELS_DIR, f"models_adapted_{ts}.pkl")
        with open(path, "wb") as f:
            pickle.dump(models, f)
        print(f"💾 {updated} model(s) updated → {path}")
    return models


# =============================================================================
# Detection + Visualization
# =============================================================================

def _apply_persistence(frame, raw_anom):
    frame = frame.copy()
    frame["anom_raw"] = raw_anom.astype(int)
    frame.sort_values(["object_id", "timestamp"], inplace=True)
    streak = (
        frame.groupby("object_id")["anom_raw"]
        .rolling(PERSIST_WINDOW, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .values
    )
    return (streak >= PERSIST_MIN).astype(int)


def detect_and_visualize(df, models, bounds):
    timestamps = sorted(df["timestamp"].dropna().unique())
    print(f"\n🕒 Processing {len(timestamps)} frames...\n")
    plt.ion()
    total_anomalies = 0

    try:
        for loc, subset in df.groupby("location"):
            if loc not in models:
                print(f"⚠️ No model for {loc}, skipping.")
                continue

            model = models[loc]["model"]
            scaler = models[loc]["scaler"]
            features = models[loc]["features"]
            cut = models[loc]["cut"]

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_title(f"{loc} — Anomaly Detection (red = anomaly)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True)
            scatter = ax.scatter([], [], c=[], alpha=0.8, s=36)
            text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

            subset = subset.sort_values(["timestamp", "object_id"])

            for t in timestamps:
                # ✅ Detect if window manually closed
                if not plt.fignum_exists(fig.number):
                    print("\n🟡 Plot window closed by user — exiting visualization early.")
                    raise KeyboardInterrupt

                frame = subset[subset["timestamp"] == t].copy()
                if len(frame) < MIN_FRAME_POINTS:
                    continue

                X = align_features(frame, features).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
                Xs = scaler.transform(X)
                scores = model.decision_function(Xs)
                raw_anom = (scores <= cut).astype(int)
                persistent = _apply_persistence(frame, raw_anom)

                near = np.zeros(len(frame), dtype=bool)
                if {"nn_dist_m", "heading_diff_deg", "ttc_s"} <= set(frame.columns):
                    nn = frame["nn_dist_m"].to_numpy(float)
                    hd = frame["heading_diff_deg"].to_numpy(float)
                    ttc = frame["ttc_s"].to_numpy(float)
                    near = (nn < NEAR_DIST_M) & (hd < NEAR_HEADING_DIFF) & (ttc < NEAR_TTC_S)

                final_anom = (persistent == 1) | ((raw_anom == 1) & near)
                num_anom = int(final_anom.sum())
                total_anomalies += num_anom

                text.set_text(
                    f"Timestamp: {pd.to_datetime(t)}\n"
                    f"Anomalies: {num_anom}/{len(frame)}"
                )

                colors = np.where(final_anom, "red", "gray")
                scatter.set_offsets(np.c_[frame["lon_scaled"], frame["lat_scaled"]])
                scatter.set_color(colors)
                plt.pause(PAUSE_SEC)

            plt.close(fig)

    except KeyboardInterrupt:
        print("\n🟡 Visualization interrupted by user (Ctrl-C or window close).")

    except Exception as e:
        print(f"\n⚠️ Visualization stopped unexpectedly: {type(e).__name__} — {e}")

    finally:
        try:
            plt.close("all")
            plt.ioff()
        except Exception:
            pass
        print("\n✅ Visualization ended gracefully.")
        print(f"📊 TOTAL anomalies detected across all frames: {total_anomalies}")




# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "existing"):
        print("Usage: python3 incident_detection.py [train|existing] [optional_csv_path]")
        sys.exit(1)

    mode = sys.argv[1]
    csv_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_CSV

    df = load_data(csv_path)
    df, bounds = scale_per_location(df)

    if mode == "train":
        print("\n🚦 Treating all data as NORMAL baseline traffic.")
        models = train_by_location(df)
    else:
        models = load_latest_models()

    detect_and_visualize(df, models, bounds)


if __name__ == "__main__":
    main()
