import numpy as np

# =========================================================
# Near-miss (interaction) thresholds
# =========================================================
NEAR_DIST_M = 8.0
NEAR_HEADING_DIFF = 45.0
NEAR_TTC_S = 3.0

PERSIST_WINDOW = 3
PERSIST_MIN = 2

CUTOFF_QUANTILE = 0.01


# =========================================================
# Collision (accident) thresholds
# =========================================================
COLLISION_PREV_SPEED_MIN = 5.0     # m/s
COLLISION_SPEED_MAX = 0.5          # m/s
COLLISION_ACCEL_MIN = -10.0        # m/s^2
COLLISION_JERK_MIN = 20.0          # m/s^3


# =========================================================
# Feature sanitization (CRITICAL for sklearn)
# =========================================================
def _sanitize_features(X):
    """
    Replace inf / -inf / NaN with finite values so sklearn won't crash.
    """
    X = np.array(X, dtype=np.float64)

    # Cap infinities (e.g., ttc_s = inf)
    X[np.isposinf(X)] = 1e3
    X[np.isneginf(X)] = -1e3

    # Replace NaNs
    X = np.nan_to_num(X, nan=0.0)

    return X


# =========================================================
# Persistence helper (near-miss only)
# =========================================================
def _apply_persistence(frame, raw_anom):
    frame = frame.copy()
    frame["anom_raw"] = raw_anom.astype(int)

    streak = (
        frame.groupby("object_id")["anom_raw"]
        .rolling(PERSIST_WINDOW, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    frame["final_anom"] = streak >= PERSIST_MIN
    return frame


# =========================================================
# Collision detection (physics-based, deterministic)
# =========================================================
def _detect_collisions(df):
    """
    Detect terminal collision events based on abrupt dynamics.
    """
    incidents = []

    df = df.sort_values(["object_id", "timestamp"])
    df["prev_speed"] = df.groupby("object_id")["speed_mps"].shift(1)

    collision_mask = (
        (df["prev_speed"] > COLLISION_PREV_SPEED_MIN) &
        (df["speed_mps"] < COLLISION_SPEED_MAX) &
        (df["accel"] < COLLISION_ACCEL_MIN) &
        (df["jerk"].abs() > COLLISION_JERK_MIN)
    )

    for (_, row) in df[collision_mask].iterrows():
        incidents.append({
            "timestamp": row["timestamp"].isoformat(),
            "location": row["location"],
            "incident_type": "collision",
            "severity": 1.0,
            "vehicles": [int(row["object_id"])],
        })

    return incidents


# =========================================================
# Near-miss detection (anomaly + physical gating)
# =========================================================
def _detect_near_misses(df, models):
    incidents = []

    for loc, subset in df.groupby("location"):
        if loc not in models:
            continue

        model = models[loc]
        features = model["features"]
        scaler = model["scaler"]
        clf = model["model"]

        # ---- Extract & sanitize features
        X = subset[features].values
        X = _sanitize_features(X)

        # ---- Scale & score
        Xs = scaler.transform(X)
        scores = clf.score_samples(Xs)

        cutoff = np.quantile(scores, CUTOFF_QUANTILE)
        raw_anom = scores < cutoff

        subset = _apply_persistence(subset, raw_anom)

        # ---- Physical gating (THIS prevents junk anomalies)
        physical_mask = (
            (subset["ttc_s"] < NEAR_TTC_S) &
            (subset["nn_dist_m"] < NEAR_DIST_M) &
            (subset["closing_rate_mps"] > 0) &
            (subset["heading_diff_deg"].abs() > NEAR_HEADING_DIFF)
        )

        subset["final_anom"] = subset["final_anom"] & physical_mask

        for ts, frame in subset.groupby("timestamp"):
            if not frame["final_anom"].any():
                continue

            incidents.append({
                "timestamp": ts.isoformat(),
                "location": loc,
                "incident_type": "near_miss",
                "severity": float(
                    min(1.0, 1.0 / max(0.1, frame["ttc_s"].min()))
                ),
                "vehicles": frame.loc[
                    frame["final_anom"], "object_id"
                ].astype(int).tolist(),
            })

    return incidents


# =========================================================
# Public API
# =========================================================
def detect_incidents(df, models):
    """
    Main incident detection entry point.

    Priority:
      1. Collisions (terminal anomalies)
      2. Near-misses (interaction anomalies)
    """

    incidents = []

    # ---- 1. Collisions FIRST
    incidents.extend(_detect_collisions(df))

    # ---- 2. Near-misses
    incidents.extend(_detect_near_misses(df, models))

    return incidents
