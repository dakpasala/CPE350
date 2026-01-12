#!/usr/bin/env python3
"""
FFMPEG METADATA XML STREAM PARSER (FINAL CLEAN VERSION + FRAME SKIPPING)

- Parses Bosch XML metadata from a file (output1.xml for now) or a stream
- Tracks multiple timestamps per ObjectId
- Preserves frame-by-frame updates internally
- Inserts into MongoDB *every Nth frame* (frame skipping)
- Uses CameraObject attributes directly
"""

import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List
import sys
import os

from camera_object import CameraObject
from pointSearch import whichLane, setLanePairsFromDBList
from collectData import pushObjectData
from mongointerface import add_count_mongo, get_camera_data
from broadcastlatlon import connect_to_server, send_websocket_data

# -------------------------------------------------------
# GLOBALS (unchanged)
# -------------------------------------------------------

speedFactor = 2.237  # m/s → mph
FRAME_SKIP = 5   # save 1 out of every 5 frames

activeRoadObjects: Dict[str, CameraObject] = {}
recentQueue: List[CameraObject] = []

currentBin = {
    "counts": defaultdict(lambda: defaultdict(int)),
    "speeds": defaultdict(lambda: defaultdict(float)),
    "timestamp": 0,
    "heatmap": {}
}

total_heatmaps: list = []
frameObjects: List[CameraObject] = []
coordinateSet: list = []

timestamp = None
openObject = False
currentObject: CameraObject | None = None
frame_counter = 0
lanes = None
camera_info = None


# -------------------------------------------------------
# PARSE LOGIC (unchanged)
# -------------------------------------------------------
def parse_element(event, elem):
    global timestamp, openObject, currentObject
    global frameObjects, coordinateSet
    global frame_counter, lanes

    tag = elem.tag.split("}")[-1]

    if tag == "Frame":
        if event == "start":
            raw_time = elem.attrib.get("UtcTime", "")
            timestamp_str = raw_time.strip()
            if timestamp_str.endswith("Z"):
                timestamp_str = timestamp_str.replace("Z", "+00:00")
            timestamp = timestamp_str

        elif event == "end":
            frame_counter += 1

            if frame_counter % FRAME_SKIP == 0:
                if frameObjects:
                    try:
                        # send_websocket_data(coordinateSet, camera_info["name"])
                        coordinateSet = []
                    except Exception as e:
                        print("Websocket error:", e)

                    pushObjectData(
                        frameObjects,
                        camera_info["name"],
                        data_push_function=add_count_mongo,
                        activeRoadObjects=activeRoadObjects,
                        recentQueue=recentQueue,
                        currentBin=currentBin,
                        total_heatmaps=total_heatmaps,
                    )

            frameObjects = []
            return

    if tag == "Object":
        if event == "start":
            openObject = True
            oid = elem.attrib.get("ObjectId")
            currentObject = CameraObject(oid, timestamp)

        elif event == "end":
            openObject = False

            if currentObject is not None:
                coord = currentObject.getCurrentLocation() if hasattr(currentObject, "getCurrentLocation") else None
                zone = currentObject.getCurrentZone() if hasattr(currentObject, "getCurrentZone") else None
                obj_type = getattr(currentObject, "detectedType", None)

                coordinateSet.append({
                    "xy": coord,
                    "zone": zone,
                    "type": obj_type,
                })
                frameObjects.append(currentObject)

            currentObject = None
            elem.clear()
            return

    if openObject and currentObject is not None:
        if tag == "GeoLocation":
            lat_str = elem.attrib.get("lat")
            lon_str = elem.attrib.get("lon")
            if lat_str and lon_str:
                try:
                    lat = float(lat_str) + float(camera_info["coordinates"][0])
                    lon = float(lon_str) + float(camera_info["coordinates"][1])
                    currentObject.setLatLon(lat, lon)

                    lane = whichLane((lat, lon), lanes)
                    currentObject.add_lane(lane)
                except Exception as e:
                    print(f"⚠ GeoLocation parse failed ({timestamp}): {e}")

        elif tag == "Type":
            if elem.text and "Likelihood" in elem.attrib:
                try:
                    currentObject.setDetectedType(elem.text)
                    currentObject.setDetectionCertainty(float(elem.attrib["Likelihood"]))
                except Exception as e:
                    print(f"⚠ Type parse failed ({timestamp}): {e}")

        elif tag == "Speed":
            if elem.text:
                try:
                    speed_mps = float(elem.text.strip())
                    currentObject.setSpeed(speed_mps * speedFactor)
                except Exception as e:
                    print(f"⚠ Speed parse failed ({timestamp}): {e}")


# -------------------------------------------------------
# ENTRYPOINT FUNCTION (NEW)
# -------------------------------------------------------
def start_ingest(camera_name: str, xml_path: str):
    global lanes, camera_info

    camera_info = get_camera_data(camera_name)
    lanes = setLanePairsFromDBList(camera_info["zones"])

    connect_to_server(8001)

    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    parser = ET.XMLPullParser(['start', 'end'])
    parser.feed("<root>")

    chunk_count = 0

    with open(xml_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            chunk_count += 1
            try:
                parser.feed(chunk)
                for event, elem in parser.read_events():
                    try:
                        parse_element(event, elem)
                    except Exception as e:
                        print("Parse error:", e)
                    finally:
                        elem.clear()
            except ET.ParseError:
                print("⚠ XML chunk malformed — recovering...")
                parser = ET.XMLPullParser(['start', 'end'])
                parser.feed("<root>")
                continue

    print(f"\n✅ Finished parsing {chunk_count} chunks")
    print(f"📉 Frame skipping enabled: saved 1 out of every {FRAME_SKIP} frames")


# -------------------------------------------------------
# SCRIPT MODE (unchanged behavior)
# -------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python ffmpegreader.py <camera_name> <xml_path>")
        sys.exit(1)

    start_ingest(sys.argv[1], sys.argv[2])
