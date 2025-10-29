#!/usr/bin/env python3
"""
incident_detection.py
Per-location IsolationForest with motion + interaction features.
Adds persistence + multi-vehicle corroboration during visualization.

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
# 0) Config
# =============================================================================

DEFAULT_CSV = "combined_vehicle_stats_expanded.csv"
MODELS_DIR = "models"
CUTOFF_QUANTILE = 0.00     # bottom 1% considered anomalous (per location)
PERSIST_WINDOW = 3         # per-vehicle streak window
PERSIST_MIN = 2            # require >=2 anomalies in last 3 for persistence
CONF_THRESHOLD = 0.5       # use if your CSV has certainty column (already used upstream)
MIN_FRAME_POINTS = 5
PAUSE_SEC = 0.35           # viz pace


# =============================================================================
# 1) Data loading + scaling
# =============================================================================

def load_data(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    # ensure timestamp is datetime and sorted
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values(["timestamp"]).reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows from {csv_path}")
    return df


def scale_per_location(df: pd.DataFrame):
    """Add lat_scaled / lon_scaled in [0,1] per location (for plotting)."""
    df = df.copy()
    bounds = {}
    for loc, g in df.groupby("location"):
        lat_min, lat_max = g["lat"].min(), g["lat"].max()
        lon_min, lon_max = g["lon"].min(), g["lon"].max()
        # avoid div by zero
        lat_rng = (lat_max - lat_min) if lat_max != lat_min else 1.0
        lon_rng = (lon_max - lon_min) if lon_max != lon_min else 1.0

        idx = g.index
        df.loc[idx, "lat_scaled"] = (g["lat"] - lat_min) / lat_rng
        df.loc[idx, "lon_scaled"] = (g["lon"] - lon_min) / lon_rng

        bounds[loc] = (lon_min, lon_max, lat_min, lat_max)
        print(f"📍 {loc}: lat({lat_min:.6f}–{lat_max:.6f}), lon({lon_min:.6f}–{lon_max:.6f})")
    return df, bounds


# =============================================================================
# 2) Feature selection helpers
# =============================================================================

def _present(df: pd.DataFrame, cols):
    return [c for c in cols if c in df.columns]

def get_feature_columns(df: pd.DataFrame):
    """
    Choose enriched features if present. Falls back gracefully if some
    columns are missing.
    """
    # Motion & interaction features from your new CSV
    primary = [
        "speed_mps", "accel", "jerk",
        "d_heading_deg", "zone_change", "path_gap",
        "nn_dist_m", "rel_speed_mps", "heading_diff_deg",
        "closing_rate_mps", "ttc_s",
        # add scaled spatial for context (helps IF a bit):
        "lat_scaled", "lon_scaled"
    ]
    feats = _present(df, primary)

    # If none of the enriched ones exist, fallback to earlier trio
    if not feats:
        print("⚠️ Enriched features not found; falling back to ['speed','lat_scaled','lon_scaled'].")
        base = _present(df, ["speed", "lat_scaled", "lon_scaled"])
        return base

    return feats


# =============================================================================
# 3) Train / Save and Load models (per location)
# =============================================================================

def train_by_location(df: pd.DataFrame):
    """
    Train an IsolationForest per location on selected features.
    Calibrate a score cutoff (quantile) per location.
    Saves a single .pkl containing a dict keyed by location.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = {}

    features = get_feature_columns(df)
    if not features:
        raise RuntimeError("No usable features found to train on.")

    print("\n🧠 Training per-location IsolationForest models...")
    for loc, g in df.groupby("location"):
        g_train = g.copy()

        # Optional: use only confident points if available
        if "is_confident" in g_train.columns:
            g_train = g_train[g_train["is_confident"].fillna(False) == True]

        X = g_train[features].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
        if len(X) < 100:  # not enough data to fit reliably
            print(f"   • {loc}: skipped (only {len(X)} rows).")
            continue

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        model = IsolationForest(
            n_estimators=400,
            contamination=0.001,    # rough prior; we'll also use a quantile cutoff
            random_state=42,
            n_jobs=-1
        ).fit(Xs)

        # score calibration (higher score = more normal)
        scores = model.decision_function(Xs)
        cut = np.quantile(scores, CUTOFF_QUANTILE)

        models[loc] = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "cut": float(cut),
        }
        print(f"   • {loc}: trained on {len(X)} rows, cutoff (q={CUTOFF_QUANTILE:.2%})={cut:.5f}")

    # Save one file with all locations
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = os.path.join(MODELS_DIR, f"models_by_loc_{ts}.pkl")
    with open(path, "wb") as f:
        pickle.dump(models, f)

    print(f"\n💾 Saved per-location models → {path}")
    return models


def load_latest_models():
    """Load the most recent per-location model dict from /models."""
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"❌ Models directory not found: {MODELS_DIR}")

    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError("❌ No saved models found in 'models/' — please train first.")

    latest = max(files, key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)))
    path = os.path.join(MODELS_DIR, latest)

    with open(path, "rb") as f:
        models = pickle.load(f)

    print(f"📂 Loaded existing per-location models from {path}")
    return models


# =============================================================================
# 4) Detection + Visualization
# =============================================================================

def _apply_persistence(frame: pd.DataFrame, raw_anom: np.ndarray) -> np.ndarray:
    """
    Convert raw pointwise anomaly (0/1) into a persistent flag per object_id:
    require >= PERSIST_MIN anomalies in last PERSIST_WINDOW frames.
    """
    frame = frame.copy()
    frame["anom_raw"] = raw_anom.astype(int)
    frame.sort_values(["object_id", "timestamp"], inplace=True)

    # rolling sum over last k for each object
    streak = (
        frame.groupby("object_id")["anom_raw"]
        .rolling(PERSIST_WINDOW, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .values
    )
    return (streak >= PERSIST_MIN).astype(int)


def detect_and_visualize(df: pd.DataFrame, models: dict, bounds: dict):
    """
    For each location:
      - score points per frame using that location's model
      - persistent anomaly filter per object
      - optional multi-vehicle corroboration (nn_dist/heading/ttc) to mark candidates
      - live scatter (scaled lon/lat) with red anomalies
    """
    timestamps = sorted(df["timestamp"].dropna().unique())
    print(f"\n🕒 Processing {len(timestamps)} frames...\n")

    plt.ion()
    total_anomalies = 0

    try:
        for loc, subset in df.groupby("location"):
            if loc not in models:
                print(f"⚠️ No model for location '{loc}' — skipping.")
                continue

            model = models[loc]["model"]
            scaler = models[loc]["scaler"]
            features = models[loc]["features"]
            cut = models[loc]["cut"]

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.set_title(f"Live Anomaly Detection — {loc}\n(red = persistent anomaly, gray = normal)")
            ax.set_xlabel("Longitude (scaled)")
            ax.set_ylabel("Latitude (scaled)")
            ax.grid(True)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

            scatter = ax.scatter([], [], c=[], alpha=0.8, s=36)
            text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")

            # track rolling counts for display
            window_size = 5
            rolling_flags = []

            # to support persistence across frames, keep a small buffer
            # (we compute persistence per frame using prior flags stored on df)
            subset = subset.sort_values(["timestamp", "object_id"]).copy()

            for t in timestamps:
                if not plt.fignum_exists(fig.number):
                    raise KeyboardInterrupt

                frame = subset[subset["timestamp"] == t].copy()
                if len(frame) < MIN_FRAME_POINTS:
                    continue

                # Prepare features; fill robustly
                X = (
                    frame[features]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(float)
                )
                Xs = scaler.transform(X)
                scores = model.decision_function(Xs)    # higher = more normal
                raw_anom = (scores <= cut).astype(int)  # 1 = anomaly

                # Persistence filter (per object_id)
                frame["raw_anom"] = raw_anom
                # Merge a tiny history: previous frame flags to make rolling work per frame slice
                # Simpler approach: compute persistence within the current frame only,
                # approximated by previous streak stored in a dict. For clarity and robustness,
                # we recompute using small per-object buffers attached to subset itself.
                # Here, we do a simpler per-frame rolling using the helper (stateless per call).
                persistent = _apply_persistence(frame, raw_anom)

                # Multi-vehicle corroboration (if interaction cols exist)
                near = np.zeros(len(frame), dtype=bool)
                if {"nn_dist_m","heading_diff_deg","ttc_s"} <= set(frame.columns):
                    nn = frame["nn_dist_m"].to_numpy(float)
                    hd = frame["heading_diff_deg"].to_numpy(float)
                    ttc = frame["ttc_s"].to_numpy(float)
                    near = (nn < 8.0) & (hd < 45.0) & (ttc < 3.0)  # tune as needed

                # final anomaly = persistent OR (raw & corroborated)
                final_anom = (persistent == 1) | ((raw_anom == 1) & near.astype(int))

                num_anom = int(final_anom.sum())
                total_anomalies += num_anom

                rolling_flags.append(num_anom)
                if len(rolling_flags) > window_size:
                    rolling_flags.pop(0)
                avg_anom = np.mean(rolling_flags)

                # Text HUD
                text.set_text(
                    f"Timestamp: {pd.to_datetime(t)}\n"
                    f"Frame anomalies: {num_anom}/{len(frame)}\n"
                    f"Rolling avg (last {window_size}): {avg_anom:.2f}"
                )

                # Colors: red = anomaly, gray = normal
                colors = np.where(final_anom, "red", "gray")
                scatter.set_offsets(np.c_[frame["lon_scaled"], frame["lat_scaled"]])
                scatter.set_color(colors)

                plt.pause(PAUSE_SEC)

            plt.close(fig)

    except KeyboardInterrupt:
        print("\n🛑 Visualization interrupted by user (window closed).")
    finally:
        plt.ioff()
        print(f"\n📊 TOTAL persistent/corroborated anomalies across all frames: {total_anomalies}")


# =============================================================================
# 5) Main
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
        models = train_by_location(df)
    else:
        models = load_latest_models()

    detect_and_visualize(df, models, bounds)


if __name__ == "__main__":
    main()
