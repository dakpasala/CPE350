#!/usr/bin/env python3
"""
csv_tail_to_combined_stats.py

Utility script to take the LAST N rows of an old combined_stats CSV
and insert them into the MongoDB `combined_stats` collection.

Intended for quick bootstrapping / testing.
"""

import configparser
import pymongo
import pandas as pd


# =========================
# CONFIG (edit here only)
# =========================

CSV_PATH = "/Users/dakshesh/CPE 350/bosch-metadata-reader/combined_vehicle_stats_expandedNEW.csv"
TAIL_ROWS = 14
CHUNK_SIZE = 1000   # kept for consistency / future-proofing


# =========================
# Mongo connection
# =========================

def get_combined_stats_collection():
    config = configparser.ConfigParser()
    config.read("connection.ini")

    client = pymongo.MongoClient(config["DEFAULT"]["database"])
    db = client["camera-counts"]

    return db["combined_stats"]


# =========================
# CSV → Mongo
# =========================

def insert_csv_tail():
    coll = get_combined_stats_collection()

    # ✅ Ensure indexes exist
    ensure_indexes(coll)

    print(f"📄 Reading CSV: {CSV_PATH}")
    print(f"🔎 Taking last {TAIL_ROWS} rows")

    df = pd.read_csv(CSV_PATH).tail(TAIL_ROWS)

    if df.empty:
        print("⚠️ CSV is empty — nothing to insert")
        return

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    records = df.where(pd.notnull(df), None).to_dict("records")

    coll.insert_many(records)

    print(f"✅ Inserted {len(records)} documents into combined_stats")


def ensure_indexes(coll):
    """
    Create indexes if they do not already exist.
    Safe to call multiple times.
    """
    coll.create_index([("timestamp", -1)])
    coll.create_index([("location", 1)])



# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    insert_csv_tail()