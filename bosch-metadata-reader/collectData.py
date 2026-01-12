#!/usr/bin/env python3
"""
collectData.py — CLEAN REWRITE

Responsible for:
  • Receiving CameraObject instances from ffmpegreader
  • Generating per-frame documents
  • Tracking objects across frames
  • Storing EACH FRAME in MongoDB
  • Maintaining lane counts, speeds, heatmaps, active objects

This version:
  • Uses composite _id = "<camera_object_id>_<iso_timestamp>"
  • Does NOT wait for an object to disappear to push data
  • Keeps the activeRoadObjects / recentQueue tracking for future analytics
"""

from typing import Dict, List, Union
from camera_object import CameraObject

# Maximum number of historical objects for heatmap/history
RECENT_QUEUE_LIMIT = 200


def build_frame_document(obj: Union[CameraObject, dict], location: str) -> dict:
    """
    Convert a CameraObject (or old-style dict) into a MongoDB-ready document.
    Ensures every frame produces a unique _id.
    """

    # --- Case 1: new pipeline using CameraObject instances ---
    if isinstance(obj, CameraObject):
        ts = obj.timestamp
        ts_str = ts.isoformat()

        doc = {
            "id": obj.id,
            "timestamp": ts,
            "time_elapsed": 0,
            "detected_type": obj.detectedType,
            "detection_certainty": obj.detectionCertainty,
            "zones": obj.zoneHistory,
            "speed": obj.speed,
            "mapPath": obj.mapPath,
            "location": location,
        }

        # One document per (object, frame)
        doc["_id"] = f"{obj.id}_{ts_str}"
        return doc

    # --- Case 2: backwards-compatible dict pipeline ---
    if isinstance(obj, dict):
        doc = dict(obj)  # shallow copy
        doc["location"] = location

        ts = doc.get("timestamp")
        if hasattr(ts, "isoformat"):
            ts_str = ts.isoformat()
        else:
            ts_str = str(ts)

        if "id" in doc and "_id" not in doc:
            doc["_id"] = f"{doc['id']}_{ts_str}"

        return doc

    raise TypeError(f"Unsupported object type in build_frame_document: {type(obj)}")


def pushObjectData(
    objects: List[CameraObject],
    location: str,
    data_push_function,
    activeRoadObjects: Dict[str, CameraObject],
    recentQueue: List[CameraObject],
    currentBin,
    total_heatmaps
):
    """
    Main ingestion step. Called once per frame.

    For each object in the frame:
      • Generate unique per-frame document
      • Push to MongoDB immediately via data_push_function
      • Update tracking (activeRoadObjects)
      • Maintain a bounded recentQueue for analytics / heatmaps
    """

    # 1. Insert each object (per-frame) into DB
    for obj in objects:
        obj_id = str(obj.id)

        try:
            doc = build_frame_document(obj, location)
            # This calls add_count_mongo(roadObjectData, total_heatmaps, currentBin)
            data_push_function(doc, total_heatmaps, currentBin)
        except Exception as e:
            # Don't kill the entire stream if one object is bad
            print(f"⚠️ Data push failed for object {obj_id}: {e}")
            continue

        # 2. Tracking: update activeRoadObjects
        if obj_id not in activeRoadObjects:
            activeRoadObjects[obj_id] = obj
        else:
            # Merge new measurement into existing tracked object
            activeRoadObjects[obj_id].add_data(obj)

        # Mark that we saw this object in this frame
        activeRoadObjects[obj_id].modified = 1

        # 3. Add to recentQueue for analytics / heatmaps
        recentQueue.append(obj)

    # 4. Prune the recentQueue to keep it bounded
    while len(recentQueue) > RECENT_QUEUE_LIMIT:
        recentQueue.pop(0)

    # 5. Cleanup: remove objects not seen in this frame
    to_remove = []
    for obj_id, tracked in activeRoadObjects.items():
        if tracked.modified == 0:
            # Not updated this frame → object left the scene
            to_remove.append(obj_id)
        else:
            # Reset flag for next frame
            tracked.modified = 0

    for obj_id in to_remove:
        activeRoadObjects.pop(obj_id, None)
        
