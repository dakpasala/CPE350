from typing import Dict
from camera_object import CameraObject

# Two data structures:
# One with the active objects(data existed in the last push, data exists in this push)
# One with past objects(FIFO queue), upon the addition of a new object, the last object is pushed to the database
recentQueueThreshold = 20

'''
    data_push_function: The function to send the data to a database. Defaults to the standard function
    To the data push, it returns a function containing:
        - location
        - start time
        - time elapsed
        - object type
        - object certainty
        - zone #
        - speed
'''
def pushObjectData(objects, location, data_push_function, activeRoadObjects, recentQueue, currentBin, total_heatmaps):
    for roadObject in objects:
        searchID = str(roadObject["id"]) if type(roadObject) == dict else roadObject.id

        # Check to see if the objects already exist
        if searchID in activeRoadObjects:
            # If they do, update their data
            activeRoadObjects[searchID].add_data(roadObject)
        # If they don't,
        else:
            # First check the recent queue to see if they are there. If they are, re-add them to the active queue and update their data
            if searchID in recentQueue:
                objectIndex = recentQueue.index(searchID)
                returningObject = recentQueue.pop(objectIndex)

                returningObject.add_data(roadObject)

                if activeRoadObjects[searchID] != None:
                    raise ValueError
                activeRoadObjects[searchID] = returningObject
            # Otherwise, create a new instance
            else:
                # TODO: update
                newObject = roadObject if type(roadObject) == CameraObject else CameraObject(searchID, roadObject["timestamp"], roadObject["type"], roadObject["lane"], roadObject["speed"], roadObject["idle"])
                activeRoadObjects[searchID] = newObject
            

    # Grab all of the objects that were not modified this update, move them to the past queue
    objectsToPush = []
    for roadObject in activeRoadObjects:
        if activeRoadObjects[roadObject].modified == 0:
            objectsToPush.append(roadObject)
        else: activeRoadObjects[roadObject].modified = 0

    for roadObject in objectsToPush:
        # If the past queue is full, move the oldest ones onto the database
        tempObject = activeRoadObjects.pop(roadObject)
        recentQueue.append(tempObject)


    while len(recentQueue) > recentQueueThreshold:
        objectToAddToDB = recentQueue.pop(0)
        roadObjectData = objectToAddToDB.get_data()
        roadObjectData["location"] = location
        data_push_function(roadObjectData, total_heatmaps, currentBin)