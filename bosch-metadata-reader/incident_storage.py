#!/usr/bin/env python3
"""
incident_storage.py

Handles storage and retrieval of detected incidents in MongoDB.
"""

import configparser
import pymongo
from datetime import datetime
from typing import List, Dict
import pandas as pd


# =========================
# Mongo connection
# =========================

def _get_db():
    config = configparser.ConfigParser()
    config.read("connection.ini")
    client = pymongo.MongoClient(config["DEFAULT"]["database"])
    return client["camera-counts"]


def get_incidents_collection():
    return _get_db()["incidents"]


# =========================
# Save incidents
# =========================

def save_incidents(incidents: List[Dict]) -> int:
    """
    Saves detected incidents to MongoDB.
    
    Args:
        incidents: List of incident dicts from detect_incidents()
        
    Returns:
        Number of incidents saved
    """
    if not incidents:
        return 0
    
    coll = get_incidents_collection()
    
    # Enrich each incident with metadata
    enriched = []
    for incident in incidents:
        doc = incident.copy()
        
        # Add processing metadata
        doc["detected_at"] = datetime.utcnow()
        
        # Convert timestamp string to datetime if needed
        if isinstance(doc.get("timestamp"), str):
            doc["timestamp"] = pd.to_datetime(doc["timestamp"], errors="coerce")
        
        # Ensure vehicles is a list
        if "vehicles" not in doc:
            doc["vehicles"] = []
        
        enriched.append(doc)
    
    # Insert all incidents
    try:
        result = coll.insert_many(enriched, ordered=False)
        return len(result.inserted_ids)
    except pymongo.errors.BulkWriteError as e:
        # Some may have been inserted before error
        return e.details.get("nInserted", 0)


# =========================
# Query incidents
# =========================

def get_recent_incidents(
    limit: int = 100,
    location: str | None = None,
    incident_type: str | None = None,
    min_severity: float | None = None
) -> List[Dict]:
    """
    Retrieves recent incidents from MongoDB.
    
    Args:
        limit: Maximum number of incidents to return
        location: Filter by location (e.g., "patterson")
        incident_type: Filter by type ("collision" or "near_miss")
        min_severity: Minimum severity threshold (0.0 - 1.0)
        
    Returns:
        List of incident documents
    """
    coll = get_incidents_collection()
    
    query = {}
    
    if location is not None:
        query["location"] = location
    
    if incident_type is not None:
        query["incident_type"] = incident_type
    
    if min_severity is not None:
        query["severity"] = {"$gte": min_severity}
    
    cursor = (
        coll.find(query)
        .sort("timestamp", -1)
        .limit(limit)
    )
    
    incidents = []
    for doc in cursor:
        # Convert ObjectId to string for JSON serialization
        doc["_id"] = str(doc["_id"])
        incidents.append(doc)
    
    return incidents


def get_incidents_by_timerange(
    start_time: datetime,
    end_time: datetime,
    location: str | None = None
) -> List[Dict]:
    """
    Retrieves incidents within a specific time range.
    
    Args:
        start_time: Start of time range
        end_time: End of time range
        location: Optional location filter
        
    Returns:
        List of incident documents
    """
    coll = get_incidents_collection()
    
    query = {
        "timestamp": {
            "$gte": start_time,
            "$lte": end_time
        }
    }
    
    if location is not None:
        query["location"] = location
    
    cursor = coll.find(query).sort("timestamp", 1)
    
    incidents = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        incidents.append(doc)
    
    return incidents


def get_incident_statistics(location: str | None = None) -> Dict:
    """
    Get summary statistics about incidents.
    
    Args:
        location: Optional location filter
        
    Returns:
        Dict with incident counts and severity stats
    """
    coll = get_incidents_collection()
    
    query = {}
    if location is not None:
        query["location"] = location
    
    pipeline = [
        {"$match": query},
        {
            "$group": {
                "_id": "$incident_type",
                "count": {"$sum": 1},
                "avg_severity": {"$avg": "$severity"},
                "max_severity": {"$max": "$severity"}
            }
        }
    ]
    
    results = list(coll.aggregate(pipeline))
    
    stats = {
        "total": coll.count_documents(query),
        "by_type": {}
    }
    
    for result in results:
        incident_type = result["_id"]
        stats["by_type"][incident_type] = {
            "count": result["count"],
            "avg_severity": round(result["avg_severity"], 3),
            "max_severity": round(result["max_severity"], 3)
        }
    
    return stats


# =========================
# Cleanup
# =========================

def delete_old_incidents(days_old: int = 30) -> int:
    """
    Deletes incidents older than specified days.
    
    Args:
        days_old: Delete incidents older than this many days
        
    Returns:
        Number of incidents deleted
    """
    coll = get_incidents_collection()
    
    cutoff = datetime.utcnow() - pd.Timedelta(days=days_old)
    
    result = coll.delete_many({
        "timestamp": {"$lt": cutoff}
    })
    
    return result.deleted_count


# =========================
# Indexing (run once on setup)
# =========================

def create_indexes():
    """
    Creates indexes for efficient querying.
    Should be run once during initial setup.
    """
    coll = get_incidents_collection()
    
    # Timestamp index (for time range queries)
    coll.create_index([("timestamp", -1)])
    
    # Location index (for location filtering)
    coll.create_index([("location", 1)])
    
    # Type index (for filtering by incident type)
    coll.create_index([("incident_type", 1)])
    
    # Compound index for common queries
    coll.create_index([
        ("location", 1),
        ("timestamp", -1)
    ])
    
    print("✅ Incident collection indexes created")


# =========================
# CLI for testing
# =========================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Create indexes")
    parser.add_argument("--stats", action="store_true", help="Show statistics")
    parser.add_argument("--recent", type=int, help="Show N recent incidents")
    parser.add_argument("--location", type=str, help="Filter by location")
    
    args = parser.parse_args()
    
    if args.setup:
        create_indexes()
    
    if args.stats:
        stats = get_incident_statistics(location=args.location)
        print(f"\n📊 Incident Statistics")
        print(f"   Total: {stats['total']}")
        for itype, data in stats["by_type"].items():
            print(f"\n   {itype}:")
            print(f"      Count: {data['count']}")
            print(f"      Avg Severity: {data['avg_severity']}")
            print(f"      Max Severity: {data['max_severity']}")
    
    if args.recent:
        incidents = get_recent_incidents(limit=args.recent, location=args.location)
        print(f"\n📋 Recent Incidents ({len(incidents)}):")
        for inc in incidents:
            print(f"\n   {inc['timestamp']} | {inc['incident_type']} | {inc['location']}")
            print(f"      Severity: {inc['severity']:.2f}")
            print(f"      Vehicles: {inc['vehicles']}")