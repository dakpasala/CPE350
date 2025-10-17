from datetime import datetime

# Camera object class, stores and manages data for individual objects coming off of the camera
class CameraObject():
    def __init__(self, id, timestamp, boundingBox = None, centerOfGravity = None, detectedType = "None", detectionCertainty = 0.0, speed = None, objectCenter = None):
        # Unique id from the metadata
        self.id = id
        # Timestamp configuration based on the type of timestamp given
        if type(timestamp) == str:
            self.timestamp = datetime.fromisoformat(timestamp)
        elif type(timestamp) == datetime:
            self.timestamp = timestamp
        # TODO: Include support for other timestamp types?
        else: raise ValueError

        # The number of updates to the object's attributes: necessary for computing running averages
        self.numberOfUpdates = 1
        # A modified bit used for the queue of recently seen camera objects. 
        # Once this is 0 for long enough, the program pushes the info to the database.
        self.modified = 1
        # The path of the object along the camera view as an array of xy coordinates.
        # These values range from -1 to 1
        self.path = []

        # This is currently not implemented
        self.timeElapsed = 0

        # The history of zones that the object has appeared in. NOTE: This was preiously called lane history,
        # the two might still be used interchangeably in some parts of the code
        self.zoneHistory = []

        # The object's speed, in mph
        self.speed = speed
  
        if boundingBox != None:
            self.set_bounding_box_xml(boundingBox)
        else:
            # Bottom, top, right, left
            self.boundingBox = (0,0,0,0)

        if centerOfGravity != None:
            self.set_centerofgravity_xml(centerOfGravity)
        else:
            self.centerOfGravity = (0,0)

        # The detected type of the object: Car, Truck, Bike, etc.
        self.detectedType = detectedType
        self.detectionCertainty = detectionCertainty

        # Latitude longitude - differs from path as path shows traversal across the screen
        if objectCenter != None and objectCenter[0] != None and objectCenter[1] != None:
            self.mapPath = [objectCenter]
        else:
            self.mapPath = []

    '''
        Merge two camera objects together to combine their data. Accumulate the data points, average out speeds.
        NOTE: Not all values are merged, only the ones that are relevant to the database. This includes:
            Object Type and Certainty, Time Elapsed, Speed, Zone History, and Path
        NOTE: Assumes that the new object only has a number of updates equal to 1, as in it only has the immediate camera data
    '''
    def merge_object(self, newObject: 'CameraObject'):
        self.numberOfUpdates += 1
        # TODO: object type replacement - right now it gets the latest becuase the camera ususally gets the type right later on
        self.detectedType = newObject.detectedType
        self.detectionCertainty = self.get_running_average(self.detectionCertainty, newObject.detectionCertainty)

        # TODO: This is not working - fix?
        self.timeElapsed = newObject.timestamp - self.timestamp

        self.speed = self.get_running_average(self.speed, newObject.speed)

        self.combine_zone_history(newObject.zoneHistory)

        self.path += newObject.path
        # TODO: Breaking bug
        if len(self.mapPath) == 0:
            self.mapPath = newObject.mapPath
        
        if len(newObject.mapPath) > 0 and self.mapPath[-1] != newObject.mapPath[0]:
            self.mapPath += newObject.mapPath

    '''
        Uses the number of updates to get a running average of a value
    '''
    def get_running_average(self, oldValue, newValue):
        if oldValue == None: return newValue
        if newValue == None: return oldValue
        return (oldValue * ((self.numberOfUpdates-1)/self.numberOfUpdates)) + (newValue / self.numberOfUpdates)
    
    '''
        XML Functions: Parse the xml to get the object's data
    '''
    def set_bounding_box_xml(self, boundingBoxObject):
        self.boundingBox = (float(boundingBoxObject.get("bottom")), float(boundingBoxObject.get("top")), float(boundingBoxObject.get("right")), float(boundingBoxObject.get("left")))
    
    def set_centerofgravity_xml(self, centerOfGravityObject):
        self.centerOfGravity = (float(centerOfGravityObject.get("x")), float(centerOfGravityObject.get("y")))
        self.path.append(self.centerOfGravity)

    def setDetectedType(self, detectionType):
        self.detectedType = detectionType
    
    def getDetectedType(self):
        return self.detectedType
    
    def setDetectionCertainty(self, certainty):
        self.detectionCertainty = certainty
    
    def getDetectionCertainty(self):
        return self.detectionCertainty

    def setSpeed(self, speedMph):
        self.speed = speedMph

    def getSpeed(self):
        return self.speed
    
    def setLatLon(self, lat, lon):
        self.mapPath.append((lat,lon))
    
    def getCurrentLocation(self):
        if len(self.mapPath) == 0:
            return None
        return self.mapPath[-1]
    
    def getCurrentZone(self):
        if len(self.zoneHistory) == 0:
            return None
        return self.zoneHistory[-1]
    # Zone Criteria
    # If there is no zone history, the new zone is added to the list
    # If there is zone history but it is only "unknown", "unknown" is removed and the new value is added
    # If there is other zone history and the zone being added is not in the list and is also not unknown, the value is added
    def add_lane(self, zone):
        if zone != "Unknown" and zone not in self.zoneHistory:
            self.zoneHistory.append(zone)

    '''
        Used for merging objects
    '''
    def combine_zone_history(self, otherHistory):
        for zone in otherHistory:
            if zone != "Unknown" and zone not in self.zoneHistory:
                self.zoneHistory.append(zone)

    '''
        Add object data to the current object. See merge_object,
        It will almost always run merge_object except in old sections of the code
    '''
    def add_data(self, objectData):
        self.numberOfUpdates += 1
        self.modified = 1
        if type(objectData) == dict:
            self.detectedType = self.get_running_average(self.detectedType, objectData["type"])
            self.add_lane(objectData["lane"])
            self.speed = self.get_running_average(self.speed, objectData["speed"])
        elif type(objectData) == CameraObject:
            self.merge_object(objectData)

    '''
        Return all database data in a dictionary
    '''
    def get_data(self) -> dict:
        dataDict = {}
        dataDict["timestamp"] = self.timestamp
        dataDict["time_elapsed"] = 0
        dataDict["detected_type"] = self.detectedType
        dataDict["detection_certainty"] = self.detectionCertainty
        dataDict["zones"] = self.zoneHistory
        dataDict["speed"] = self.speed
        dataDict["mapPath"] = self.mapPath
        return dataDict
        

    def __str__(self):
        return f"{self.id}: {self.timestamp}, {self.detectedType}, zones {self.zoneHistory}, {self.speed}, Updated {self.numberOfUpdates} times"

