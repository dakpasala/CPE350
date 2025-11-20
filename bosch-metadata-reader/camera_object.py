from datetime import datetime

class CameraObject():
    def __init__(self, id, timestamp, boundingBox=None, centerOfGravity=None,
                 detectedType="None", detectionCertainty=0.0, speed=None, objectCenter=None):

        # Unique object ID from Bosch
        self.id = str(id)

        # Normalize timestamp → always ISO string + keep datetime version
        if isinstance(timestamp, str):
            self.timestamp_str = timestamp
            self.timestamp = datetime.fromisoformat(timestamp)
        elif isinstance(timestamp, datetime):
            self.timestamp = timestamp
            self.timestamp_str = timestamp.isoformat()
        else:
            raise ValueError("Timestamp must be str or datetime")

        # Tracking helpers
        self.numberOfUpdates = 1
        self.modified = 1

        # Path in normalized screen coords
        self.path = []

        # GPS path (lat/lon)
        self.mapPath = []
        if objectCenter and objectCenter[0] and objectCenter[1]:
            self.mapPath.append(objectCenter)

        # Zones
        self.zoneHistory = []

        # Speed (mph)
        self.speed = speed

        # Object type + confidence
        self.detectedType = detectedType
        self.detectionCertainty = detectionCertainty

        # Bounding box
        if boundingBox is not None:
            self.set_bounding_box_xml(boundingBox)
        else:
            self.boundingBox = (0, 0, 0, 0)

        # Center of gravity
        if centerOfGravity is not None:
            self.set_centerofgravity_xml(centerOfGravity)
        else:
            self.centerOfGravity = (0, 0)

        # Time elapsed between updates
        self.timeElapsed = 0


    # ---------------- XML data updates ---------------- #

    def set_bounding_box_xml(self, boundingBoxObject):
        self.boundingBox = (
            float(boundingBoxObject.get("bottom")),
            float(boundingBoxObject.get("top")),
            float(boundingBoxObject.get("right")),
            float(boundingBoxObject.get("left")),
        )

    def set_centerofgravity_xml(self, centerOfGravityObject):
        self.centerOfGravity = (
            float(centerOfGravityObject.get("x")),
            float(centerOfGravityObject.get("y")),
        )
        self.path.append(self.centerOfGravity)

    def setDetectedType(self, detectionType):
        self.detectedType = detectionType

    def setDetectionCertainty(self, certainty):
        self.detectionCertainty = certainty

    def setSpeed(self, speedMph):
        self.speed = speedMph

    def setLatLon(self, lat, lon):
        self.mapPath.append((lat, lon))

    def add_lane(self, zone):
        if zone != "Unknown" and zone not in self.zoneHistory:
            self.zoneHistory.append(zone)
    # --- Compatibility with old ffmpegreader --- #
    def getCurrentLocation(self):
        """Return last (lat, lon) for backward compatibility."""
        if len(self.mapPath) == 0:
            return None
        return self.mapPath[-1]

    def getCurrentZone(self):
        """Return most recent zone for backward compatibility."""
        if len(self.zoneHistory) == 0:
            return None
        return self.zoneHistory[-1]



    # ---------------- Merging logic ---------------- #

    def get_running_average(self, oldValue, newValue):
        if oldValue is None:
            return newValue
        if newValue is None:
            return oldValue
        return (oldValue * ((self.numberOfUpdates - 1) / self.numberOfUpdates)) + \
               (newValue / self.numberOfUpdates)

    def merge_object(self, newObject: 'CameraObject'):
        self.numberOfUpdates += 1

        # Keep newest type (Bosch gets more accurate over time)
        self.detectedType = newObject.detectedType
        self.detectionCertainty = self.get_running_average(
            self.detectionCertainty, newObject.detectionCertainty
        )

        # delta time
        self.timeElapsed = (newObject.timestamp - self.timestamp).total_seconds()

        # speed
        self.speed = self.get_running_average(self.speed, newObject.speed)

        # zones
        for z in newObject.zoneHistory:
            self.add_lane(z)

        # map path
        if len(newObject.mapPath) > 0:
            if not self.mapPath or self.mapPath[-1] != newObject.mapPath[0]:
                self.mapPath.extend(newObject.mapPath)

        # screen path
        self.path.extend(newObject.path)


    def add_data(self, objectData):
        """Merge dict or CameraObject into current object."""
        self.numberOfUpdates += 1
        self.modified = 1

        if isinstance(objectData, dict):
            self.detectedType = objectData.get("detected_type", self.detectedType)
            self.detectionCertainty = objectData.get(
                "detection_certainty", self.detectionCertainty
            )
            self.speed = objectData.get("speed", self.speed)
            lane = objectData.get("zone")
            if lane:
                self.add_lane(lane)

        elif isinstance(objectData, CameraObject):
            self.merge_object(objectData)


    # ---------------- Export for DB ---------------- #

    def get_data(self) -> dict:
        dataDict = {}
        # include the raw object id so downstream code can use it
        dataDict["id"] = self.id

        # timestamp and kinematics
        dataDict["timestamp"] = self.timestamp
        dataDict["time_elapsed"] = 0

        # classification
        dataDict["detected_type"] = self.detectedType
        dataDict["detection_certainty"] = self.detectionCertainty

        # zones, speed, path
        dataDict["zones"] = self.zoneHistory
        dataDict["speed"] = self.speed
        dataDict["mapPath"] = self.mapPath

        return dataDict


    def __str__(self):
        return f"{self.id}: {self.timestamp_str}, {self.detectedType}, zones {self.zoneHistory}, speed {self.speed}, updates {self.numberOfUpdates}"
