#!/usr/bin/env python3
"""
auto_retrain.py

Automatically train location-specific models for ANY camera location.
Creates separate .pkl files per location (e.g., patterson.pkl, foothill.pkl).
"""

import os
import sys
import pickle
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Add parent directory to path to import from root-level data.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data import load_all_combined_stats  # ← Import from root data.py


# =========================
# Config
# =========================

MODELS_DIR = "../models"  # Go up one level from incident_detection/ to models/
RETRAIN_THRESHOLD = 1000  # Minimum samples needed to train
MIN_RETRAIN_INTERVAL_HOURS = 6  # Don't retrain more often than this


# =========================
# Model loading
# =========================

def load_model_for_location(location):
    """Load model for a specific location if it exists."""
    model_path = os.path.join(MODELS_DIR, f"{location}.pkl")
    
    if not os.path.exists(model_path):
        return None
    
    try:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    except:
        return None


def load_all_models():
    """Load all location models into a dict."""
    models = {}
    
    if not os.path.exists(MODELS_DIR):
        return models
    
    for filename in os.listdir(MODELS_DIR):
        if filename.endswith(".pkl"):
            location = filename.replace(".pkl", "")
            model = load_model_for_location(location)
            if model:
                models[location] = model
    
    return models


# =========================
# Feature columns
# =========================

def get_feature_columns(df):
    """Get available features from dataframe."""
    primary = [
        "speed_mps", "accel", "jerk",
        "heading_diff_deg", "zone_change", "path_gap",
        "nn_dist_m", "rel_speed_mps",
        "closing_rate_mps", "ttc_s",
        "lat_scaled", "lon_scaled",
    ]
    
    # Only use columns that exist
    feats = [c for c in primary if c in df.columns]
    
    # Fallback
    if not feats or len(feats) < 3:
        feats = [c for c in ["speed_mps", "accel", "lat_scaled", "lon_scaled"] 
                if c in df.columns]
    
    return feats


def scale_per_location(df):
    """Scale lat/lon to 0-1 range."""
    df = df.copy()
    
    lat_min, lat_max = df["lat"].min(), df["lat"].max()
    lon_min, lon_max = df["lon"].min(), df["lon"].max()
    
    lat_rng = (lat_max - lat_min) or 1.0
    lon_rng = (lon_max - lon_min) or 1.0
    
    df["lat_scaled"] = (df["lat"] - lat_min) / lat_rng
    df["lon_scaled"] = (df["lon"] - lon_min) / lon_rng
    
    return df


# =========================
# Training
# =========================

def train_model_for_location(location, force=False):
    """
    Train a new Isolation Forest model for ANY location.
    
    Args:
        location: Camera location name (e.g., "patterson", "foothill")
        force: Force retrain even if model exists
    
    Returns:
        bool: True if trained successfully
    """
    
    print(f"\n{'='*60}")
    print(f"🧠 TRAINING MODEL FOR: {location.upper()}")
    print(f"{'='*60}\n")
    
    # Check if model already exists
    model_path = os.path.join(MODELS_DIR, f"{location}.pkl")
    
    if os.path.exists(model_path) and not force:
        # Check age of model
        model_age_hours = (datetime.now().timestamp() - os.path.getmtime(model_path)) / 3600
        
        if model_age_hours < MIN_RETRAIN_INTERVAL_HOURS:
            print(f"✅ Model already exists and is recent ({model_age_hours:.1f}h old)")
            print(f"   Skipping retrain (minimum interval: {MIN_RETRAIN_INTERVAL_HOURS}h)")
            return False
    
    # Load data for this location
    print(f"[1/5] Loading {location} data from MongoDB...")
    df = load_all_combined_stats(
        time_range="month",  # Last 30 days
        limit=50000,
        location=location
    )
    
    if df.empty:
        print(f"❌ No data found for {location}")
        return False
    
    print(f"✅ Loaded {len(df)} samples")
    
    # Check threshold
    if len(df) < RETRAIN_THRESHOLD:
        print(f"⚠️  Not enough data yet ({len(df)} < {RETRAIN_THRESHOLD})")
        print(f"   Need {RETRAIN_THRESHOLD - len(df)} more samples")
        return False
    
    # Scale lat/lon
    print(f"\n[2/5] Scaling coordinates...")
    df = scale_per_location(df)
    
    # Get features
    print(f"\n[3/5] Preparing features...")
    features = get_feature_columns(df)
    print(f"✅ Using features: {features}")
    
    # Extract feature matrix
    X = (
        df[features]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(float)
    )
    
    print(f"✅ Feature matrix: {X.shape}")
    
    # Fit scaler
    print(f"\n[4/5] Training model...")
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    
    # Train Isolation Forest
    model = IsolationForest(
        n_estimators=400,
        contamination=0.001,  # Expect 0.1% anomalies
        random_state=42,
        n_jobs=-1,
        verbose=0
    ).fit(Xs)
    
    # Calculate cutoff
    scores = model.decision_function(Xs)
    cutoff = float(np.quantile(scores, 0.01))
    
    print(f"✅ Model trained")
    
    # Save model
    print(f"\n[5/5] Saving model...")
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    model_data = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "cut": cutoff,
    }
    
    # Backup old model if exists
    if os.path.exists(model_path):
        backup_path = model_path.replace(".pkl", f"_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        os.rename(model_path, backup_path)
        print(f"📦 Backed up old model to {os.path.basename(backup_path)}")
    
    # Save new model
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Saved to {model_path}")
    
    # Stats
    anomalies = (scores < cutoff).sum()
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   Location: {location}")
    print(f"   Samples: {len(X)}")
    print(f"   Features: {len(features)}")
    print(f"   Anomalies detected: {anomalies} ({anomalies/len(X)*100:.2f}%)")
    print(f"   Cutoff score: {cutoff:.4f}")
    
    print(f"\n{'='*60}")
    print(f"✅ {location.upper()} MODEL READY!")
    print(f"{'='*60}\n")
    
    return True


# =========================
# Auto-training on data arrival
# =========================

def check_and_train_for_location(location):
    """
    Check if location has enough data and train if needed.
    Call this from server.py when processing data.
    """
    
    # Check if model exists
    model = load_model_for_location(location)
    
    if model is not None:
        # Model exists - check if it needs updating
        model_path = os.path.join(MODELS_DIR, f"{location}.pkl")
        model_age_hours = (datetime.now().timestamp() - os.path.getmtime(model_path)) / 3600
        
        if model_age_hours < MIN_RETRAIN_INTERVAL_HOURS:
            return False  # Too soon to retrain
    
    # Check data count
    from data import get_db
    db = get_db()
    count = db.combined_stats.count_documents({"location": location})
    
    if count < RETRAIN_THRESHOLD:
        if model is None:
            print(f"⏳ [{location}] Waiting for more data ({count}/{RETRAIN_THRESHOLD})")
        return False
    
    # Train!
    print(f"\n🚀 [{location}] Auto-training triggered ({count} samples available)")
    return train_model_for_location(location)


# =========================
# Main
# =========================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python auto_retrain.py <location> [--force]")
        print("Example: python auto_retrain.py patterson")
        print("Example: python auto_retrain.py foothill --force")
        sys.exit(1)
    
    location = sys.argv[1]
    force = "--force" in sys.argv
    
    print(f"🚀 Training model for: {location}")
    
    success = train_model_for_location(location, force=force)
    
    if success:
        print(f"\n✅ Success! {location} model is now active.")
    else:
        print(f"\n❌ Training failed - check requirements above.")