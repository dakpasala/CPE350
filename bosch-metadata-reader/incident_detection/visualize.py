import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# Config (same values as original)
# =============================================================================

MIN_FRAME_POINTS = 5
PAUSE_SEC = 0.35

NEAR_DIST_M = 2.0
NEAR_HEADING_DIFF = 45.0
NEAR_TTC_S = 3.0

WINDOW_SIZE = 20
WINDOW_STEP = 5
WINDOW_MIN_FRAMES = 10

PERSIST_WINDOW = 3
PERSIST_MIN = 2


# =============================================================================
# Helpers (unchanged)
# =============================================================================

def _apply_persistence(frame, raw_anom):
    frame = frame.copy()
    frame["anom_raw"] = raw_anom.astype(int)

    streak = (
        frame.groupby("object_id")["anom_raw"]
        .rolling(PERSIST_WINDOW, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
        .values
    )

    return (streak >= PERSIST_MIN).astype(int)


def group_near_miss_events(frame):
    events = []
    if len(frame) < 2:
        return events

    anom_idx = frame.index[frame["final_anom"] == True].tolist()
    if not anom_idx:
        return events

    for i in range(len(anom_idx)):
        for j in range(i + 1, len(anom_idx)):
            a, b = frame.loc[anom_idx[i]], frame.loc[anom_idx[j]]

            if (
                abs(a["lon_scaled"] - b["lon_scaled"]) < 0.02
                and abs(a["lat_scaled"] - b["lat_scaled"]) < 0.02
                and a.get("nn_dist_m", 99) < NEAR_DIST_M
                and b.get("nn_dist_m", 99) < NEAR_DIST_M
            ):
                events.append((a["object_id"], b["object_id"]))

    return events


def log_detected_events(events, timestamp, location):
    if not events:
        return

    out_file = "detected_events.csv"
    header = ["timestamp", "location", "object_id_A", "object_id_B"]
    file_exists = os.path.exists(out_file)

    with open(out_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        for a, b in events:
            writer.writerow([timestamp, location, a, b])


def detect_window_anomalies(
    timestamps,
    frame_anomaly_counts,
    window_size=WINDOW_SIZE,
    step=WINDOW_STEP,
    min_frames=WINDOW_MIN_FRAMES,
):
    ts_sorted = sorted(timestamps)
    window_events = []

    if len(ts_sorted) < window_size:
        return window_events

    for i in range(0, len(ts_sorted) - window_size + 1, step):
        window_ts = ts_sorted[i : i + window_size]
        frames_with_anoms = sum(
            1 for t in window_ts if frame_anomaly_counts.get(t, 0) > 0
        )

        if frames_with_anoms >= min_frames:
            window_events.append((window_ts[0], window_ts[-1], frames_with_anoms))

    return window_events


def log_window_events(
    location,
    window_events,
    window_size=WINDOW_SIZE,
    step=WINDOW_STEP,
    min_frames=WINDOW_MIN_FRAMES,
):
    if not window_events:
        return

    out_file = "window_anomalies.csv"
    header = [
        "location",
        "start_timestamp",
        "end_timestamp",
        "frames_with_anomalies",
        "window_size",
        "step",
        "min_frames",
    ]

    file_exists = os.path.exists(out_file)

    with open(out_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)

        for start_ts, end_ts, frames_with_anoms in window_events:
            writer.writerow([
                location,
                start_ts,
                end_ts,
                frames_with_anoms,
                window_size,
                step,
                min_frames,
            ])


# =============================================================================
# Visualization runner (UNCHANGED behavior)
# =============================================================================

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

            frame_anomaly_counts = {}

            for t in timestamps:
                if not plt.fignum_exists(fig.number):
                    raise KeyboardInterrupt

                frame = subset[subset["timestamp"] == t].copy()
                if len(frame) < MIN_FRAME_POINTS:
                    continue

                X = (
                    frame[features]
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                    .to_numpy(float)
                )

                Xs = scaler.transform(X)
                scores = model.decision_function(Xs)
                raw_anom = (scores <= cut).astype(int)
                persistent = _apply_persistence(frame, raw_anom)

                near = np.zeros(len(frame), dtype=bool)
                if {"nn_dist_m", "heading_diff_deg", "ttc_s"} <= set(frame.columns):
                    nn = frame["nn_dist_m"].to_numpy(float)
                    hd = frame["heading_diff_deg"].to_numpy(float)
                    ttc = frame["ttc_s"].to_numpy(float)
                    near = (
                        (nn < NEAR_DIST_M)
                        & (hd < NEAR_HEADING_DIFF)
                        & (ttc < NEAR_TTC_S)
                        & (frame["closing_rate_mps"] < -0.5)
                    )


                final_anom = (persistent == 1) | near
                # final_anom = (persistent == 1) | ((raw_anom == 1) & near)

                num_anom = int(final_anom.sum())
                total_anomalies += num_anom

                frame["final_anom"] = final_anom
                frame_anomaly_counts[t] = num_anom

                events = group_near_miss_events(frame)
                if events:
                    print(f"🚨 {len(events)} event(s) at {t} in {loc}: {events}")
                    log_detected_events(events, t, loc)

                text.set_text(
                    f"Timestamp: {pd.to_datetime(t)}\n"
                    f"Anomalies: {num_anom}/{len(frame)}"
                )

                colors = np.where(final_anom, "red", "gray")
                scatter.set_offsets(np.c_[frame["lon_scaled"], frame["lat_scaled"]])
                scatter.set_color(colors)
                plt.pause(PAUSE_SEC)

            if frame_anomaly_counts:
                window_events = detect_window_anomalies(
                    timestamps=list(frame_anomaly_counts.keys()),
                    frame_anomaly_counts=frame_anomaly_counts,
                )
                log_window_events(loc, window_events)

            plt.close(fig)

    except KeyboardInterrupt:
        print("\n🟡 Visualization interrupted by user.")

    except Exception as e:
        print(f"\n⚠️ Visualization stopped unexpectedly: {type(e).__name__} — {e}")

    finally:
        plt.close("all")
        plt.ioff()
        print("\n✅ Visualization ended gracefully.")
        print(f"📊 TOTAL anomalies detected across all frames: {total_anomalies}")
