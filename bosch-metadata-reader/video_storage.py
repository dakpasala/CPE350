"""
video_storage.py

Handles video storage and retrieval using MongoDB GridFS.
"""

import pymongo
import gridfs
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import configparser


# =========================
# MongoDB connection
# =========================

def _get_db():
    """Get MongoDB database connection"""
    config = configparser.ConfigParser()
    config.read("connection.ini")
    client = pymongo.MongoClient(config["DEFAULT"]["database"])
    return client["camera-counts"]


def get_gridfs():
    """Get GridFS instance for video storage"""
    db = _get_db()
    return gridfs.GridFS(db)


def get_videos_collection():
    """Get videos metadata collection"""
    return _get_db()["videos"]


# =========================
# Save video to GridFS
# =========================

def save_video_to_gridfs(
    file_content: bytes,
    filename: str,
    camera: str,
    timestamp: str,
    duration: int
) -> str:
    """
    Saves video file to GridFS and metadata to videos collection.
    
    Args:
        file_content: Raw video file bytes
        filename: Original filename (e.g., "patterson_20260206_143022.mp4")
        camera: Camera location name
        timestamp: Timestamp string (e.g., "20260206_143022")
        duration: Video duration in seconds
    
    Returns:
        video_id: String ID of the saved video
    """
    fs = get_gridfs()
    videos_coll = get_videos_collection()
    
    # Parse timestamp
    try:
        video_timestamp = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
    except ValueError:
        video_timestamp = datetime.utcnow()
    
    # Store video in GridFS
    video_id = fs.put(
        file_content,
        filename=filename,
        content_type="video/mp4",
        camera=camera,
        timestamp=video_timestamp,
        duration=duration,
        upload_date=datetime.utcnow()
    )
    
    # Store metadata in videos collection for easy querying
    video_metadata = {
        "_id": video_id,
        "filename": filename,
        "camera": camera,
        "timestamp": video_timestamp,
        "duration": duration,
        "upload_date": datetime.utcnow(),
        "size_bytes": len(file_content),
        "incident_ids": []  # Will be populated when incidents detected
    }
    
    videos_coll.insert_one(video_metadata)
    
    print(f"✅ Video saved to GridFS: {video_id} | {filename} | {len(file_content) / 1024 / 1024:.2f} MB")
    
    return str(video_id)


# =========================
# Retrieve video from GridFS
# =========================

def get_video_by_id(video_id: str) -> Optional[gridfs.GridOut]:
    """
    Retrieves video file from GridFS by ID.
    
    Args:
        video_id: String ID of the video
    
    Returns:
        GridFS file object or None if not found
    """
    from bson import ObjectId
    
    fs = get_gridfs()
    
    try:
        oid = ObjectId(video_id)
        return fs.get(oid)
    except Exception as e:
        print(f"⚠️ Video not found: {video_id} | {e}")
        return None


# =========================
# Query videos
# =========================

def get_recent_videos(
    limit: int = 50,
    camera: Optional[str] = None
) -> List[Dict]:
    """
    Get recent videos metadata.
    
    Args:
        limit: Max number of videos to return
        camera: Filter by camera location
    
    Returns:
        List of video metadata dictionaries
    """
    videos_coll = get_videos_collection()
    
    query = {}
    if camera:
        query["camera"] = camera
    
    cursor = videos_coll.find(query).sort("timestamp", -1).limit(limit)
    
    videos = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])  # Convert ObjectId to string
        videos.append(doc)
    
    return videos


def get_videos_by_timerange(
    start_time: datetime,
    end_time: datetime,
    camera: Optional[str] = None
) -> List[Dict]:
    """
    Get videos within a time range.
    
    Args:
        start_time: Start of time range
        end_time: End of time range
        camera: Filter by camera location
    
    Returns:
        List of video metadata dictionaries
    """
    videos_coll = get_videos_collection()
    
    query = {
        "timestamp": {
            "$gte": start_time,
            "$lte": end_time
        }
    }
    
    if camera:
        query["camera"] = camera
    
    cursor = videos_coll.find(query).sort("timestamp", 1)
    
    videos = []
    for doc in cursor:
        doc["_id"] = str(doc["_id"])
        videos.append(doc)
    
    return videos


def get_video_for_incident(incident_timestamp: datetime, camera: str) -> Optional[Dict]:
    """
    Find video that covers a specific incident timestamp.
    
    Args:
        incident_timestamp: When the incident occurred
        camera: Camera location
    
    Returns:
        Video metadata dict or None
    """
    videos_coll = get_videos_collection()
    
    # Find video where incident_timestamp falls within [timestamp, timestamp + duration]
    query = {
        "camera": camera,
        "timestamp": {"$lte": incident_timestamp}
    }
    
    # Get all videos that started before the incident
    cursor = videos_coll.find(query).sort("timestamp", -1)
    
    for video in cursor:
        video_start = video["timestamp"]
        video_end = video_start + timedelta(seconds=video["duration"])
        
        # Check if incident falls within this video's time range
        if video_start <= incident_timestamp <= video_end:
            video["_id"] = str(video["_id"])
            return video
    
    return None


def link_video_to_incident(video_id: str, incident_id: str):
    """
    Links a video to an incident by adding incident_id to video metadata.
    
    Args:
        video_id: Video ID
        incident_id: Incident ID to link
    """
    from bson import ObjectId
    
    videos_coll = get_videos_collection()
    
    try:
        videos_coll.update_one(
            {"_id": ObjectId(video_id)},
            {"$addToSet": {"incident_ids": incident_id}}
        )
        print(f"✅ Linked video {video_id} to incident {incident_id}")
    except Exception as e:
        print(f"⚠️ Failed to link video to incident: {e}")


# =========================
# Cleanup
# =========================

def delete_old_videos(days_old: int = 30) -> int:
    """
    Deletes videos older than N days from both GridFS and metadata collection.
    
    Args:
        days_old: Delete videos older than this many days
    
    Returns:
        Number of videos deleted
    """
    from bson import ObjectId
    
    cutoff_date = datetime.utcnow() - timedelta(days=days_old)
    
    fs = get_gridfs()
    videos_coll = get_videos_collection()
    
    # Find old videos
    old_videos = list(videos_coll.find({"timestamp": {"$lt": cutoff_date}}))
    
    deleted_count = 0
    for video in old_videos:
        try:
            # Delete from GridFS
            fs.delete(video["_id"])
            
            # Delete metadata
            videos_coll.delete_one({"_id": video["_id"]})
            
            deleted_count += 1
        except Exception as e:
            print(f"⚠️ Failed to delete video {video['_id']}: {e}")
    
    print(f"🧹 Deleted {deleted_count} old videos (older than {days_old} days)")
    
    return deleted_count