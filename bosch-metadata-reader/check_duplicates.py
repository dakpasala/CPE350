#!/usr/bin/env python3
"""
check_duplicates.py

Checks if objects have multiple observations (trajectories) in combined_stats.
"""

import configparser
import pymongo
from collections import defaultdict
import pandas as pd


# =========================
# Mongo connection
# =========================

def get_db():
    config = configparser.ConfigParser()
    config.read("connection.ini")

    client = pymongo.MongoClient(config["DEFAULT"]["database"])
    return client["camera-counts"]


def main():
    db = get_db()
    coll = db["combined_stats"]

    print("📥 Loading combined_stats…")
    cursor = coll.find(
        {},
        {"object_id": 1, "timestamp": 1, "location": 1, "accel": 1, "jerk": 1}
    )

    # Group by object_id
    object_observations = defaultdict(list)
    total_rows = 0
    
    for doc in cursor:
        obj_id = doc.get("object_id")
        timestamp = doc.get("timestamp")
        location = doc.get("location")
        accel = doc.get("accel")
        jerk = doc.get("jerk")
        
        object_observations[obj_id].append({
            "timestamp": timestamp,
            "location": location,
            "accel": accel,
            "jerk": jerk
        })
        total_rows += 1

    print(f"\n📊 Total rows: {total_rows}")
    print(f"🎯 Unique objects: {len(object_observations)}")
    
    # Analyze trajectories
    single_obs = 0
    multi_obs = 0
    trajectory_lengths = []
    
    for obj_id, observations in object_observations.items():
        count = len(observations)
        trajectory_lengths.append(count)
        
        if count == 1:
            single_obs += 1
        else:
            multi_obs += 1
    
    print(f"\n📈 Trajectory Analysis:")
    print(f"   Objects with 1 observation:  {single_obs} ({single_obs/len(object_observations)*100:.1f}%)")
    print(f"   Objects with 2+ observations: {multi_obs} ({multi_obs/len(object_observations)*100:.1f}%)")
    
    if trajectory_lengths:
        print(f"\n📏 Trajectory Length Stats:")
        print(f"   Min:     {min(trajectory_lengths)} observations")
        print(f"   Max:     {max(trajectory_lengths)} observations")
        print(f"   Average: {sum(trajectory_lengths)/len(trajectory_lengths):.1f} observations")
    
    # Show sample multi-observation objects
    if multi_obs > 0:
        print(f"\n✅ Sample objects with trajectories:")
        shown = 0
        for obj_id, observations in object_observations.items():
            if len(observations) > 1:
                observations_sorted = sorted(observations, key=lambda x: x["timestamp"])
                timestamps = [obs["timestamp"] for obs in observations_sorted]
                accels = [obs["accel"] for obs in observations_sorted]
                
                accel_populated = sum(1 for a in accels if a is not None and not pd.isna(a))
                
                print(f"\n   Object {obj_id}:")
                print(f"      Observations: {len(observations)}")
                print(f"      Time span: {timestamps[0]} → {timestamps[-1]}")
                print(f"      Accel populated: {accel_populated}/{len(observations)}")
                
                shown += 1
                if shown >= 3:
                    break
    
    # Check for accel/jerk population
    print(f"\n🔬 Feature Population Check:")
    accel_count = 0
    jerk_count = 0
    total = 0
    
    for obj_id, observations in object_observations.items():
        for obs in observations:
            total += 1
            if obs["accel"] is not None and not pd.isna(obs["accel"]):
                accel_count += 1
            if obs["jerk"] is not None and not pd.isna(obs["jerk"]):
                jerk_count += 1
    
    print(f"   Accel populated: {accel_count}/{total} ({accel_count/total*100:.1f}%)")
    print(f"   Jerk populated:  {jerk_count}/{total} ({jerk_count/total*100:.1f}%)")
    
    if accel_count == 0:
        print("\n⚠️  WARNING: No accel values found! Objects need multiple observations within same window.")
    
    if multi_obs == 0:
        print("\n⚠️  WARNING: No objects have multiple observations!")
        print("   This means each object only appears once - no trajectories are being tracked.")
        print("   Check that your 15-second window is accumulating data properly.")


if __name__ == "__main__":
    main()