import asyncio
from websockets.sync.client import connect
import datetime
import json

websocket = None
# Dictionary of coordinates. Each entry corresponds to a camera, is broken down and sent thru the websocket
coordinatesDict = {}
lastSentTime = 0
# The rate, per second, of the number of sends of location data the program should be sending
sendRateFPS = 8


def connect_to_server(portNumber):
    global websocket
    try:
        websocket = connect(f"ws://localhost:{portNumber}")
    except:
        print("Failed to connect to interface")

def is_ws_connected():
    try:
        websocket.recv()
        return websocket.connected
    except:
        return False

def send_websocket_data(data, cameraName):
    global lastSentTime, sendRate
    if websocket == None: return
    if lastSentTime == 0:
        lastSentTime = datetime.datetime.now()
    # Add the coordinates to the dictionary of cameras, replacing if necessary

    # Check the amount of time since the last send. if it's greater than the rate, send again
    timeDifference = datetime.datetime.now() - lastSentTime
    if timeDifference > datetime.timedelta(seconds= (1 / sendRateFPS)):
        # print(json.dumps(accumulate_points()))
        websocket.send(json.dumps(data))
        lastSentTime = datetime.datetime.now()
    # if (is_ws_connected()):

def accumulate_points():
    outputArray = []
    for key in coordinatesDict:
        outputArray += coordinatesDict[key]
    return outputArray