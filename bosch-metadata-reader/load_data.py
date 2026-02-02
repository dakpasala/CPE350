import configparser
import pandas as pd
import pymongo


def _get_db():
    config = configparser.ConfigParser()
    config.read("connection.ini")
    client = pymongo.MongoClient(config["DEFAULT"]["database"])
    return client["camera-counts"]


def load_all_combined_stats(limit: int = 10_000, location: str | None = None):
    db = _get_db()

    query = {}
    if location is not None:
        query["location"] = location

    cursor = (
        db["combined_stats"]
        .find(query)
        .sort("timestamp", -1)
        .limit(limit)
    )

    data = list(cursor)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).drop(columns=["_id"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_incidents(
    limit: int = 10,
    location: str | None = None,
    incident_type: str | None = None,
    min_severity: float | None = None
) -> pd.DataFrame:
    """
    Retrieve incidents from DB.

    Args:
        limit: Number of incidents to return (default 10)
        location: Filter by location (e.g. "patterson")
        incident_type: Filter by type ("collision" or "near_miss")
        min_severity: Minimum severity (0.0 - 1.0)

    Returns:
        DataFrame sorted by timestamp (newest first)
    """
    db = _get_db()

    query = {}

    if location is not None:
        query["location"] = location

    if incident_type is not None:
        query["incident_type"] = incident_type

    if min_severity is not None:
        query["severity"] = {"$gte": min_severity}

    cursor = (
        db["incidents"]
        .find(query)
        .sort("timestamp", -1)
        .limit(limit)
    )

    data = list(cursor)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data).drop(columns=["_id"], errors="ignore")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp", ascending=False).reset_index(drop=True)

    return df