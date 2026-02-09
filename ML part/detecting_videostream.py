import os

# CRITICAL: Set this BEFORE importing torch/ultralytics to fix PyTorch 2.6 compatibility
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

# Patch torch.load to use weights_only=False
import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("best.pt")

# Set the dimensions for captured frames
frame_width = 640
frame_height = 480

# Local video file path (change this to use different videos)
video_path = "../bosch-metadata-reader/videos/youtube.mp4"

# Alternative: Use live stream instead
# stream_url = "http://kamera.mikulov.cz:8888/mjpg/video.mjpg"
# cap = cv2.VideoCapture(stream_url)

# Start capturing from local video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video file at {video_path}")
    exit()

print(f"Processing video: {video_path}")
print("Press 'q' to quit")

while cap.isOpened():
    # boolean success flag and the video frame
    # if the video has ended, the success flag is False
    is_frame, frame = cap.read()
    if not is_frame:
        print("Video ended")
        break

    # Resize frame (optional - you can remove this if you want full resolution)
    resized_frame = cv2.resize(frame, (frame_width, frame_height))

    # Option 1: Direct prediction (recommended - faster, no temp files)
    results = model.predict(source=resized_frame, show=True)

    # Option 2: Save to temp file first (original approach - uncomment if needed)
    # temp_image_path = "temp/temp.jpg"
    # cv2.imwrite(temp_image_path, resized_frame)
    # results = model.predict(source=temp_image_path, show=True)

    # Check for the 'q' key press to quit
    if cv2.waitKey(1) & 0xff == ord('q'):
        print("Quit by user")
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
print("Processing complete")