"""
video_buffer.py

Temporarily stores uploaded videos until incident detection completes.
Videos are kept in a buffer and only saved to GridFS if incidents are detected.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional
import threading

# =========================
# Video buffer storage
# =========================

# Structure: {
#   "camera_timestamp": {
#       "file_content": bytes,
#       "filename": str,
#       "camera": str,
#       "timestamp": datetime,
#       "duration": int,
#       "upload_time": datetime
#   }
# }

VIDEO_BUFFER = {}
BUFFER_LOCK = threading.Lock()

# Videos older than this will be auto-deleted (safety mechanism)
MAX_BUFFER_AGE_SECONDS = 60


# =========================
# Buffer operations
# =========================

def add_video_to_buffer(
    file_content: bytes,
    filename: str,
    camera: str,
    timestamp_str: str,
    duration: int
) -> str:
    """
    Adds a video to the temporary buffer.
    
    Returns:
        buffer_key: Unique key for this video
    """
    # Parse timestamp
    try:
        video_timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
    except ValueError:
        video_timestamp = datetime.utcnow()
    
    buffer_key = f"{camera}_{timestamp_str}"
    
    with BUFFER_LOCK:
        VIDEO_BUFFER[buffer_key] = {
            "file_content": file_content,
            "filename": filename,
            "camera": camera,
            "timestamp": video_timestamp,
            "duration": duration,
            "upload_time": datetime.utcnow()
        }
    
    print(f"📦 Video buffered: {buffer_key} | {len(file_content) / 1024 / 1024:.2f} MB")
    
    return buffer_key


def get_video_from_buffer(camera: str, timestamp: datetime) -> Optional[Dict]:
    """
    Retrieves a video from buffer by camera and timestamp.
    
    Args:
        camera: Camera location
        timestamp: Video timestamp (datetime object)
    
    Returns:
        Video data dict or None
    """
    with BUFFER_LOCK:
        # Try to find video that covers this timestamp
        for key, video_data in VIDEO_BUFFER.items():
            if video_data["camera"] != camera:
                continue
            
            video_start = video_data["timestamp"]
            video_end = video_start + timedelta(seconds=video_data["duration"])
            
            # Check if timestamp falls within video window
            if video_start <= timestamp <= video_end:
                return video_data
    
    return None


def remove_video_from_buffer(buffer_key: str) -> bool:
    """
    Removes a video from the buffer.
    
    Returns:
        True if removed, False if not found
    """
    with BUFFER_LOCK:
        if buffer_key in VIDEO_BUFFER:
            del VIDEO_BUFFER[buffer_key]
            print(f"🗑️ Video removed from buffer: {buffer_key}")
            return True
    
    return False


def cleanup_old_videos():
    """
    Removes videos from buffer that are too old.
    This is a safety mechanism to prevent memory leaks.
    """
    now = datetime.utcnow()
    cutoff = now - timedelta(seconds=MAX_BUFFER_AGE_SECONDS)
    
    with BUFFER_LOCK:
        to_remove = []
        
        for key, video_data in VIDEO_BUFFER.items():
            if video_data["upload_time"] < cutoff:
                to_remove.append(key)
        
        for key in to_remove:
            del VIDEO_BUFFER[key]
            print(f"⚠️ Auto-deleted old buffered video: {key}")
        
        if to_remove:
            print(f"🧹 Cleaned up {len(to_remove)} old buffered videos")


def get_buffer_stats() -> Dict:
    """
    Returns statistics about the video buffer.
    """
    with BUFFER_LOCK:
        total_size = sum(len(v["file_content"]) for v in VIDEO_BUFFER.values())
        
        return {
            "count": len(VIDEO_BUFFER),
            "total_size_mb": total_size / 1024 / 1024,
            "videos": [
                {
                    "key": key,
                    "camera": v["camera"],
                    "timestamp": v["timestamp"].isoformat(),
                    "size_mb": len(v["file_content"]) / 1024 / 1024,
                    "age_seconds": (datetime.utcnow() - v["upload_time"]).total_seconds()
                }
                for key, v in VIDEO_BUFFER.items()
            ]
        }