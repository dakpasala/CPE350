from bs4 import BeautifulSoup
from camera_object import CameraObject
from broadcastlatlon import send_websocket_data
import json
import re

# The speeds coming off of the camera are in meters per second. Currently multiplying by this factor to convert to mph
speedFactor = 2.237

# Return the objects detected in an xml packet
def parseXml(inputData, whichLane, lanes, cameraName, offset):
    frameObjects = []
    # For the purpose of continuity, the parser is placed in a try-except block.
    # If the metadata packet is bad/not formatted correctly, it drops the packet instead of raising an error
    try:
        xmlSoup = BeautifulSoup(inputData, 'xml')
        videoFrame = xmlSoup.Frame
        if videoFrame == None: return
        timestamp = videoFrame.get('UtcTime')
        coordinateSet = []
    except Exception as error:
        print(error)
        print(xmlSoup)
        return None

    # Each object is in an <Object /> element
    for roadObject in videoFrame.find_all("Object"):
      
        # Add metadata to the detected object
        try:
            lat = None
            lon = None
            if roadObject.GeoLocation != None:
                lat = float(roadObject.GeoLocation.get("lat")) + float(offset[0])
                lon = float(roadObject.GeoLocation.get("lon")) + float(offset[1])

            objectType = ""
            detectionCertainty = 0.0
            # Search for a type
            
            if roadObject.VehicleInfo != None:
                objectType = roadObject.VehicleInfo.Type.string
                detectionCertainty = roadObject.VehicleInfo.Type.get("Likelihood")
            elif roadObject.Class != None:
                objectType = roadObject.Class.find_all("Type")[-1].string
                detectionCertainty = roadObject.Class.Likelihood.string
            speed = None
            if roadObject.Speed != None:
                speed =  float(roadObject.Speed.string) * speedFactor
            currentObject = CameraObject(
                id = roadObject.get("ObjectId"), 
                timestamp= timestamp, 
                boundingBox= roadObject.BoundingBox, 
                centerOfGravity= roadObject.CenterOfGravity, 
                detectedType= objectType, 
                detectionCertainty= float(detectionCertainty),
                speed= speed,
                objectCenter = (lat, lon)
            )
            lane = whichLane((lat, lon), lanes)
            currentObject.add_lane(lane)
            coordinateSet.append({
                "xy": (lat, lon),
                "zone": lane,
                "type": objectType
            })
            
            frameObjects.append(currentObject)
        except Exception as error:
            print("\033[31m", error)
            print("\033[33m", roadObject, "\033[0m")
            # return None
        # TODO: Make this multiprocessable: compile all coordinate data into one place and then push it every few seconds(?)
    try:
        send_websocket_data(coordinateSet, cameraName)
    except Exception as error:
        print("Coordinate Livestream Error", error)
 
    return frameObjects

    