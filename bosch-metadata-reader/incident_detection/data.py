import os
import pandas as pd

DEFAULT_CSV = "combined_vehicle_stats_expandedNEW2.csv"

def load_data(csv_path: str = DEFAULT_CSV) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    df["timestamp"] = (
        df["timestamp"]
        .astype(str)
        .str.strip()
        .apply(lambda x: x.replace("Z", "+00:00") if isinstance(x, str) else x)
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    if "heading_diff_deg" not in df.columns and "d_heading_deg" in df.columns:
        df["heading_diff_deg"] = df["d_heading_deg"]

    df = df.sort_values(["location", "timestamp"]).reset_index(drop=True)
    print(f"✅ Loaded {len(df)} rows from CSV")
    return df


def scale_per_location(df: pd.DataFrame):
    df = df.copy()
    bounds = {}

    for loc, g in df.groupby("location"):
        lat_min, lat_max = g["lat"].min(), g["lat"].max()
        lon_min, lon_max = g["lon"].min(), g["lon"].max()

        lat_rng = (lat_max - lat_min) or 1.0
        lon_rng = (lon_max - lon_min) or 1.0

        idx = g.index
        df.loc[idx, "lat_scaled"] = (g["lat"] - lat_min) / lat_rng
        df.loc[idx, "lon_scaled"] = (g["lon"] - lon_min) / lon_rng

        bounds[loc] = (lon_min, lon_max, lat_min, lat_max)

    print("📍 Scaled lat/lon per location.")
    return df, bounds