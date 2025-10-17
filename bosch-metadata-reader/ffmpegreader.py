import xml.etree.ElementTree as ET
import re
from typing import Dict
from collections import defaultdict
import sys
import subprocess

from camera_object import CameraObject
from pointSearch import whichLane, setLanePairsFromDBList
from collectData import pushObjectData
from mongointerface import add_count_mongo, get_camera_data
from broadcastlatlon import connect_to_server, send_websocket_data

# TODO: Get camera data from mongodb
camera_info = get_camera_data(sys.argv[1])
connect_to_server(8001)


# The speeds coming off of the camera are in meters per second. Currently multiplying by this factor to convert to mph
speedFactor = 2.237

# CAMERA SPECIFIC DATA TRACKING OBJECTS - The reason why these are here is that they were originally global variables. This makes the program thread-safe
# For use in collection to send to the db. For more info see collectData.py
activeRoadObjects: Dict[str, CameraObject] = {}
recentQueue: list[CameraObject] = []
# For use in the mongodb interface
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

def parse_element(event, elem):
    global timestamp, openObject, currentObject, frameObjects, camera_info, coordinateSet, lanes
    if elem.tag == "root": return
    # Look for the opening of the frame to collect objects

    tag = elem.tag.split("}")[1]

    if tag == "Frame":
        if event == "start":
            timestamp = elem.attrib['UtcTime']
        elif event == "end":
            # TODO: Send the objects and live coordinates
            if frameObjects != []:   
                try:
                    # send_websocket_data(coordinateSet, camera_info["name"])
                    coordinateSet = []
                except Exception as error:
                    print("Coordinate Livestream Error", error)
                pushObjectData(
                    frameObjects, 
                    camera_info["name"], 
                    data_push_function = add_count_mongo, 
                    activeRoadObjects=activeRoadObjects, 
                    recentQueue=recentQueue,
                    currentBin= currentBin,
                    total_heatmaps=total_heatmaps)
                frameObjects = []

            
    elif tag == "Object":
        # starting a new object
        if event == "start":
            openObject = True
            currentObject = CameraObject(elem.attrib["ObjectId"], timestamp)
            # create the new class
        # closing the object and pushing it off
        elif event == "end":
            openObject = False
            elem.clear()
            # push current object
            coordinateSet.append({
                "xy": currentObject.getCurrentLocation(),
                "zone": currentObject.getCurrentZone(),
                "type": currentObject.getDetectedType()
            })
            frameObjects.append(currentObject)
            currentObject = None
    elif openObject == True:
        if tag == "GeoLocation":
            lat = float(elem.attrib["lat"]) + float(camera_info["coordinates"][0])
            lon = float(elem.attrib["lon"]) + float(camera_info["coordinates"][1])
            currentObject.setLatLon(lat, lon)
            # TODO: Set zone area
            lane = whichLane((lat, lon), lanes)
            currentObject.add_lane(lane)
        elif tag == "Point":
            pass
        elif tag == "Type":
            if 'Likelihood' in elem.attrib and elem.text != None:
                currentObject.setDetectedType(elem.text) 
                currentObject.setDetectionCertainty(float(elem.attrib['Likelihood']))
        elif tag == "Speed":
            currentObject.setSpeed(float(elem.text) * speedFactor)
        else:
            pass
    

command = f'ffmpeg -i "oldstuff/output1.xml" -map 0:d -c copy -copy_unknown -loglevel fatal -f data -'
with subprocess.Popen(
    command, stdout=subprocess.PIPE, shell=True
) as process:
    parser = ET.XMLPullParser(['start', 'end'])
    # Add a root element to prevent the parser from complaining about bad xml
    parser.feed('<root>')

    foundStart = False
    i = 0
    while True:
        i += 1
        value = process.stdout.read1()
        # Reset to a new block
        if not foundStart:
            # Search for the beginning of the packet
            res = re.search("<tt:MetadataStream", value.decode('utf-8'))
            if res != None:
                value = value[res.start():]
                foundStart = True

        # If the beginning of the packet has been found and everything is working properly:
        if foundStart:
            # Start parsing the data
            parser.feed(value)
            # Refrain for a little bit to grab a full packet
            if i > 5:
                # Put into a try-except block as any bad xml(from lost data) raises an exception
                try:
                    for event, elem in parser.read_events():
                        parse_element(event, elem)
                        elem.clear()
                # If there is bad data, we drop the packet and reset the parser
                except:
                    i = 0
                    foundStart = False
                    parser = ET.XMLPullParser(['start', 'end'])
                    parser.feed('<root>')
        
