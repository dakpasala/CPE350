import os
import pickle
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MODELS_DIR = "models"
CUTOFF_QUANTILE = 0.01


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
        feats = _present(df, ["speed_mps", "lat_scaled", "lon_scaled"])
    return feats


def train_by_location(df):
    os.makedirs(MODELS_DIR, exist_ok=True)
    models = {}
    features = get_feature_columns(df)

    print("🧠 Training IsolationForest per location...")

    for loc, g in df.groupby("location"):
        g = g[g.get("is_confident", True).astype(bool)]

        X = (
            g[features]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
            .to_numpy(float)
        )

        if len(X) < 50:
            continue

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)

        model = IsolationForest(
            n_estimators=400,
            contamination=0.001,
            random_state=42,
            n_jobs=-1,
        ).fit(Xs)

        scores = model.decision_function(Xs)
        cut = float(np.quantile(scores, CUTOFF_QUANTILE))

        models[loc] = {
            "model": model,
            "scaler": scaler,
            "features": features,
            "cut": cut,
        }

        print(f"  • {loc}: trained on {len(X)} samples")

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out = os.path.join(MODELS_DIR, f"models_by_loc_{ts}.pkl")

    with open(out, "wb") as f:
        pickle.dump(models, f)

    print(f"💾 Models saved → {out}")
    return models


def load_latest_models():
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    if not files:
        raise FileNotFoundError("No trained models found.")

    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(MODELS_DIR, f)))
    with open(os.path.join(MODELS_DIR, latest), "rb") as f:
        return pickle.load(f)
