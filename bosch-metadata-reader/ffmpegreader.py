import xml.etree.ElementTree as ET
import re
from typing import Dict
from collections import defaultdict
import sys

from camera_object import CameraObject
from pointSearch import whichLane, setLanePairsFromDBList
from collectData import pushObjectData
from mongointerface import add_count_mongo, get_camera_data
from broadcastlatlon import connect_to_server, send_websocket_data

# -------------------------------------------------------
# 1. Initialize camera info + global vars
# -------------------------------------------------------
camera_info = get_camera_data(sys.argv[1])
connect_to_server(8001)

speedFactor = 2.237  # m/s → mph

activeRoadObjects: Dict[str, CameraObject] = {}
recentQueue: list[CameraObject] = []
currentBin = {
    "counts": defaultdict(lambda: defaultdict(int)),
    "speeds": defaultdict(lambda: defaultdict(float)),
    "timestamp": 0,
    "heatmap": {}
}
lanes = setLanePairsFromDBList(camera_info["zones"])
total_heatmaps = []
timestamp = None
openObject = False
frameObjects = []
currentObject: CameraObject | None = None
coordinateSet = []


# -------------------------------------------------------
# 2. parse_element() — identical logic, safe fixes included
# -------------------------------------------------------
def parse_element(event, elem):
    global timestamp, openObject, currentObject, frameObjects, camera_info, coordinateSet, lanes
    if elem.tag == "root":
        return

    tag = elem.tag.split("}")[-1]

    if tag == "Frame":
        if event == "start":
            raw_time = elem.attrib.get("UtcTime", "")
            timestamp = raw_time.strip().replace("Z", "+00:00")
        elif event == "end":
            if frameObjects:
                try:
                    # send_websocket_data(coordinateSet, camera_info["name"])
                    coordinateSet = []
                except Exception as error:
                    print("Coordinate Livestream Error", error)

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

    elif tag == "Object":
        if event == "start":
            openObject = True
            currentObject = CameraObject(elem.attrib["ObjectId"], timestamp)
        elif event == "end":
            openObject = False
            elem.clear()
            if currentObject is not None:
                coordinateSet.append({
                    "xy": currentObject.getCurrentLocation(),
                    "zone": currentObject.getCurrentZone(),
                    "type": currentObject.getDetectedType()
                })
                frameObjects.append(currentObject)
            currentObject = None

    elif openObject and currentObject is not None:
        if tag == "GeoLocation":
            lat_str = elem.attrib.get("lat")
            lon_str = elem.attrib.get("lon")
            if not lat_str or not lon_str:
                return
            try:
                lat = float(lat_str) + float(camera_info["coordinates"][0])
                lon = float(lon_str) + float(camera_info["coordinates"][1])
                currentObject.setLatLon(lat, lon)
                lane = whichLane((lat, lon), lanes)
                currentObject.add_lane(lane)
            except Exception as e:
                print(f"⚠️ GeoLocation parse failed at {timestamp}: {e}")

        elif tag == "Type":
            if "Likelihood" in elem.attrib and elem.text:
                currentObject.setDetectedType(elem.text)
                currentObject.setDetectionCertainty(float(elem.attrib["Likelihood"]))

        elif tag == "Speed":
            if elem.text is None:
                return
            try:
                speed_val = float(elem.text.strip()) if elem.text.strip() else 0.0
                currentObject.setSpeed(speed_val * speedFactor)
            except Exception as e:
                print(f"⚠️ Speed parse failed at {timestamp}: {e}")


# -------------------------------------------------------
# 3. Read and parse XML (robust stream-safe version)
# -------------------------------------------------------
import os, glob
import xml.etree.ElementTree as ET

def parse_single_xml(xml_path):
    print(f"\n🚗 Processing {os.path.basename(xml_path)} ...")
    parser = ET.XMLPullParser(['start', 'end'])
    parser.feed('<root>')
    processed_frames = set()
    chunk_count = 0

    with open(xml_path, "rb") as process:
        for chunk in iter(lambda: process.read(4096), b""):
            chunk_count += 1
            try:
                parser.feed(chunk)
                for event, elem in parser.read_events():
                    try:
                        parse_element(event, elem)
                        if timestamp and timestamp in processed_frames:
                            continue
                        if timestamp:
                            processed_frames.add(timestamp)
                    except Exception as e:
                        print(f"⚠️ parse_element error: {e}")
                    finally:
                        elem.clear()
            except ET.ParseError as e:
                print(f"⚠️ Skipping malformed chunk ({e})")
                parser = ET.XMLPullParser(['start', 'end'])
                parser.feed('<root>')
                continue

    print(f"✅ Finished {os.path.basename(xml_path)} — {len(processed_frames)} unique frames, {chunk_count} chunks read.")

# -------------------------------------------------------
# Loop through all XMLs in ../xmls/
# -------------------------------------------------------
xml_folder = "../xmls"
xml_files = sorted(glob.glob(os.path.join(xml_folder, "*.xml")))

if not xml_files:
    raise FileNotFoundError(f"No XML files found in {xml_folder}")

for xml_path in xml_files:
    parse_single_xml(xml_path)

print("\n🏁 All XMLs processed sequentially — Mongo updated safely.")