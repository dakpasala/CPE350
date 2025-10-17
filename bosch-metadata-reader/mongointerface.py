import pymongo
import configparser
from datetime import datetime, timedelta
from collections import defaultdict

from heatmap import add_to_heatmap, extract_heatmap

import json

config = configparser.ConfigParser()
config.read("connection.ini")
dbUrl = config["DEFAULT"]["database"]

client = pymongo.MongoClient(dbUrl)
cameraData = client["camera-counts"]
countCollection  = cameraData["counts"]
vehicleCollection = cameraData["vehicles"]
cameraCollection = cameraData["cameras"]
heatmapCollection = cameraData["heatmaps"]

def collect_heatmap(total_heatmaps, current_bin, amount_to_accumulate):
    total_heatmaps.append(current_bin["heatmap"])
    # Logic to determine if it's time to send off heatmap data to server
    if len(total_heatmaps) >= amount_to_accumulate:
        heatmap_bin = {
            "location": current_bin["location"],
            "timestamp": current_bin["timestamp"],
            "heatmap": extract_heatmap(total_heatmaps),
        }
        heatmapCollection.insert_one(heatmap_bin)
        del total_heatmaps[:]


def round_timestamp(timestamp: datetime, interval = 300):
    overTime = timestamp.timestamp() % interval
    roundedValue = timestamp.timestamp() - overTime
    timestamp = datetime.fromtimestamp(roundedValue, tz=timestamp.tzinfo)
    return timestamp

# TODO: Update this
def get_camera_data(id = None):
    if id != None:
        query = {"name": id}
        cameraInfo = cameraCollection.find_one(query)
        return cameraInfo
    cameraInfo = cameraCollection.find()
    return list(cameraInfo)


def add_countBin(location, total_heatmaps, currentBin):
    time = currentBin["timestamp"]
    counts = currentBin["counts"]
    speeds = currentBin["speeds"]
    newBin = {
        "timestamp": time,
        "location": location,
        "interval": 300,
        "counts": [],
        "speeds": [],
        "heatmap": currentBin["heatmap"]
    }
    for zone in currentBin["counts"]:
        countsObject = counts[zone]
        countsObject["zone"] = zone
        speedsObject = speeds[zone]
        speedsObject["zone"] = zone
        newBin["counts"].append(countsObject)
        newBin["speeds"].append(speedsObject)
    collect_heatmap(total_heatmaps, newBin, amount_to_accumulate=12)
    newBin.pop("heatmap")
    countCollection.insert_one(newBin)
    print(f"Added data to {location} at {datetime.now()}")

def add_count_mongo(roadObjectData, total_heatmaps, currentBin):
    if currentBin["timestamp"] == 0:
        currentBin["timestamp"] = round_timestamp(roadObjectData["timestamp"])

    upperBound = currentBin["timestamp"] + timedelta(minutes=5)
    # Check to see if this object is outside of the bin in memory
    if roadObjectData["timestamp"] >= upperBound:
        # If it is, push that bin to the database and create a new bin
        add_countBin(roadObjectData["location"], total_heatmaps, currentBin)
        currentBin["counts"] = defaultdict(lambda: defaultdict(int))
        currentBin["speeds"] = defaultdict(lambda: defaultdict(float))
        currentBin["timestamp"] = round_timestamp(roadObjectData["timestamp"])
        currentBin["heatmap"] = {}
        
    # Add this object's data to that bin
    if len(roadObjectData["zones"]) == 0:
        roadObjectData["zones"].append("Unknown")
    if roadObjectData['detected_type'] == None:
        roadObjectData['detected_type'] = "Unknown"
    for zone in roadObjectData["zones"]:
    # Increment the bin count and grab its value for the running average calculation
        currentBin["counts"][zone][roadObjectData["detected_type"]] += 1
        totalValue = currentBin["counts"][zone][roadObjectData["detected_type"]]
        averageSpeed = currentBin["speeds"][zone][roadObjectData["detected_type"]]
        currentBin["speeds"][zone][roadObjectData["detected_type"]] = get_running_average(averageSpeed, roadObjectData["speed"], totalValue)
    add_to_heatmap(currentBin["heatmap"], roadObjectData)

    vehicleCollection.insert_one(roadObjectData)


def get_running_average(oldValue, newValue, total):
    # New average = old average * (n-1)/n + new value /n
    if oldValue == None: return newValue
    if newValue == None: return oldValue
    return (oldValue * ((total - 1)/total)) + (newValue / total)
