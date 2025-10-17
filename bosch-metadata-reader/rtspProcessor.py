import subprocess
import threading
import signal
import time

from pointSearch import setLanes, whichLane, setLanePairsFromDBList
from mongointerface import add_count_mongo, get_camera_data

def stream_data(address, name, offset, whichLane, zoneCoordinates, dataPushFunction = add_count_mongo):
    # The command to access the bosch metadata
    command = f'python ffmpegreader.py {name}'

    # Fork the active process to open the command line and run ffmpeg
    with subprocess.Popen(
        command, shell=True
    ) as process:
        print(f"Starting processor for {name}...")
        # TODO: Implement a drop number
        while True:
            time.sleep(30000)
    

def runProcessorMonoProcessing():
    # Get the necessary information about the camera from the database
    camera_info = get_camera_data('dunbarton')
    # Broadcast latitude longitude data
    # connect_to_server(8001)
    # Set zone coordinates - global variables with the coordinates of the zone boundaries
    setLanePairsFromDBList(camera_info["zones"])
    # Begin streaming the data from the camera
    stream_data(camera_info["url"], camera_info["name"], whichLane, add_count_mongo)

def runProcessorMultiProcessing():
    # Get the necessary information about the camera from the database
    camera_info = get_camera_data()
    # Broadcast latitude longitude data
    # connect_to_server(8001)
    # Set zone coordinates - global variables with the coordinates of the zone boundaries
    for camera in camera_info:
        t = threading.Thread(target=stream_data, args=(camera["url"], camera["name"], camera["coordinates"], whichLane, camera["zones"], add_count_mongo))
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        t.start()

if __name__ == "__main__":
    runProcessorMultiProcessing()
 

