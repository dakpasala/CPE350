#!/usr/bin/env python3
"""
export_anomalies_for_pydeck.py
Loads CSV + saved per-location models (.pkl) and exports anomaly points to CSV
for pydeck overlay.

Output: anomaly_points.csv with cols:
timestamp, lat, lon, location, object_id, iforest_score
"""

import pickle
import numpy as np
import pandas as pd

# -----------------------------
# Config (edit these)
# -----------------------------
CSV_PATH = "combined_vehicle_stats_expandedNEW.csv"
PKL_PATH = "models_by_loc_2026-01-09_10-59.pkl"
OUT_PATH = "anomaly_points.csv"

# from teammate
PERSIST_WINDOW = 3
PERSIST_MIN = 2

NEAR_DIST_M = 8.0
NEAR_HEADING_DIFF = 45.0
NEAR_TTC_S = 3.0

MIN_FRAME_POINTS = 5  # skip tiny frames


# -----------------------------
# Helpers (short cmnts)
# -----------------------------
def load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # fix ts
    df["timestamp"] = (
        df["timestamp"].astype(str).str.strip()
        .apply(lambda x: x.replace("Z", "+00:00") if isinstance(x, str) else x)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # heading diff mismatch
    if "heading_diff_deg" not in df.columns and "d_heading_deg" in df.columns:
        df["heading_diff_deg"] = df["d_heading_deg"]

    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    print(f"loaded rows: {len(df)}")
    return df


def scale_per_location(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for loc, g in df.groupby("location"):
        lat_min, lat_max = g["lat"].min(), g["lat"].max()
        lon_min, lon_max = g["lon"].min(), g["lon"].max()

        lat_rng = (lat_max - lat_min) if lat_max != lat_min else 1.0
        lon_rng = (lon_max - lon_min) if lon_max != lon_min else 1.0

        idx = g.index
        df.loc[idx, "lat_scaled"] = (g["lat"] - lat_min) / lat_rng
        df.loc[idx, "lon_scaled"] = (g["lon"] - lon_min) / lon_rng

    return df


def align_features(frame: pd.DataFrame, model_features) -> pd.DataFrame:
    aligned = frame.copy()
    for f in model_features:
        if f not in aligned.columns:
            aligned[f] = 0.0
    return aligned[model_features]


def apply_persistence(frame: pd.DataFrame, raw_anom: np.ndarray) -> np.ndarray:
    tmp = frame.copy()
    tmp["anom_raw"] = raw_anom.astype(int)
    tmp = tmp.sort_values(["object_id", "timestamp"])

    streak = (
        tmp.groupby("object_id")["anom_raw"]
        .rolling(PERSIST_WINDOW, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .values
    )
    return (streak >= PERSIST_MIN).astype(int)


# -----------------------------
# Main
# -----------------------------
def main():
    df = load_csv(CSV_PATH)
    df = scale_per_location(df)

    print("loading pkl...")
    with open(PKL_PATH, "rb") as f:
        models = pickle.load(f)

    out_parts = []
    total_frames = 0
    total_anom_rows = 0

    for loc, subset in df.groupby("location"):
        if loc not in models:
            print(f"skip {loc}: no model")
            continue

        model = models[loc]["model"]
        scaler = models[loc]["scaler"]
        features = models[loc]["features"]
        cut = float(models[loc]["cut"])

        subset = subset.sort_values(["timestamp", "object_id"])

        # group by timestamp = "frame"
        for t, frame in subset.groupby("timestamp"):
            if pd.isna(t):
                continue
            if len(frame) < MIN_FRAME_POINTS:
                continue

            total_frames += 1

            X = (
                align_features(frame, features)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
                .to_numpy(float)
            )

            Xs = scaler.transform(X)
            scores = model.decision_function(Xs)

            preds = model.predict(Xs)  # -1 outlier, +1 inlier
            print(loc, "pred -1 count:", int((preds == -1).sum()), "/", len(preds))


            raw_anom = (scores <= cut).astype(int)
            persistent = apply_persistence(frame, raw_anom)

            # near-miss corroboration
            near = np.zeros(len(frame), dtype=bool)
            needed = {"nn_dist_m", "heading_diff_deg", "ttc_s"}
            if needed <= set(frame.columns):
                nn = frame["nn_dist_m"].to_numpy(float)
                hd = frame["heading_diff_deg"].to_numpy(float)
                ttc = frame["ttc_s"].to_numpy(float)
                near = (nn < NEAR_DIST_M) & (hd < NEAR_HEADING_DIFF) & (ttc < NEAR_TTC_S)

            final_anom = (persistent == 1) | ((raw_anom == 1) & near)

            if final_anom.any():
                fr = frame.loc[final_anom].copy()
                fr["iforest_score"] = scores[final_anom]
                fr["location"] = loc
                out_parts.append(fr[["timestamp", "lat", "lon", "location", "object_id", "iforest_score"]])
                total_anom_rows += int(final_anom.sum())

    if out_parts:
        out = pd.concat(out_parts, ignore_index=True)
    else:
        out = pd.DataFrame(columns=["timestamp", "lat", "lon", "location", "object_id", "iforest_score"])

    out.to_csv(OUT_PATH, index=False)
    print(f"wrote: {OUT_PATH}")
    print(f"frames processed: {total_frames}")
    print(f"anom rows written: {len(out)} (counted {total_anom_rows})")


if __name__ == "__main__":
    main()
