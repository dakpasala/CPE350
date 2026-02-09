"""
Video Analysis Script using YOLOv8 for Accident Detection
Analyzes Caltrans traffic camera footage for potential accidents
"""

import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import argparse
from pathlib import Path


class AccidentDetector:
    def __init__(self, model_path):
        """
        Initialize the accident detector with YOLOv8 model
        
        Args:
            model_path: Path to the YOLOv8 .pt model file
        """
        self.model = YOLO(model_path)
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck in COCO dataset
        self.tracked_vehicles = defaultdict(list)  # Store vehicle positions over time
        self.stationary_threshold = 30  # frames (about 1 second at 30fps)
        self.accidents_detected = []
        
    def is_stationary(self, track_id, current_pos, threshold=10):
        """
        Check if a vehicle has been stationary
        
        Args:
            track_id: Unique tracking ID of the vehicle
            current_pos: Current position (x, y) of vehicle center
            threshold: Pixel threshold for movement
            
        Returns:
            bool: True if vehicle is stationary
        """
        if track_id not in self.tracked_vehicles:
            return False
        
        history = self.tracked_vehicles[track_id]
        if len(history) < self.stationary_threshold:
            return False
        
        # Check if vehicle hasn't moved much in last N frames
        recent_positions = history[-self.stationary_threshold:]
        movements = [np.linalg.norm(np.array(current_pos) - np.array(pos)) 
                    for pos in recent_positions]
        
        return max(movements) < threshold
    
    def detect_accidents(self, frame, frame_number):
        """
        Run YOLO detection and identify potential accidents
        
        Args:
            frame: Video frame to analyze
            frame_number: Current frame number
            
        Returns:
            annotated_frame: Frame with bounding boxes and alerts
            detections: List of detected accidents
        """
        # Run YOLOv8 tracking
        results = self.model.track(frame, persist=True, verbose=False)
        
        annotated_frame = frame.copy()
        current_detections = []
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for box, track_id, cls, conf in zip(boxes, track_ids, classes, confidences):
                # Only process vehicles
                if cls not in self.vehicle_classes:
                    continue
                
                x1, y1, x2, y2 = box
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Store position history
                self.tracked_vehicles[track_id].append((center_x, center_y))
                
                # Check if stationary
                is_stopped = self.is_stationary(track_id, (center_x, center_y))
                
                # Draw bounding box
                color = (0, 0, 255) if is_stopped else (0, 255, 0)  # Red if stopped, green otherwise
                thickness = 3 if is_stopped else 2
                
                cv2.rectangle(annotated_frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            color, thickness)
                
                # Add label
                label = f"ID:{track_id}"
                if is_stopped:
                    label += " STOPPED!"
                    current_detections.append({
                        'frame': frame_number,
                        'track_id': track_id,
                        'position': (center_x, center_y),
                        'confidence': conf
                    })
                
                cv2.putText(annotated_frame, label,
                          (int(x1), int(y1) - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return annotated_frame, current_detections
    
    def analyze_video(self, video_path, output_path=None, show_preview=False):
        """
        Analyze a video for accidents
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save annotated video
            show_preview: Whether to show video preview during processing
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nAnalyzing: {video_path.name}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_number = 0
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            annotated_frame, detections = self.detect_accidents(frame, frame_number)
            all_detections.extend(detections)
            
            # Add frame number
            cv2.putText(annotated_frame, f"Frame: {frame_number}/{total_frames}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Write output
            if out:
                out.write(annotated_frame)
            
            # Show preview
            if show_preview:
                cv2.imshow('Accident Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_number += 1
            
            # Progress indicator
            if frame_number % 30 == 0:
                print(f"Progress: {frame_number}/{total_frames} frames", end='\r')
        
        cap.release()
        if out:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n\nAnalysis Complete!")
        print(f"Total frames processed: {frame_number}")
        print(f"Potential accidents detected: {len(all_detections)}")
        
        if all_detections:
            print("\nDetection Summary:")
            unique_incidents = defaultdict(list)
            for det in all_detections:
                unique_incidents[det['track_id']].append(det['frame'])
            
            for track_id, frames in unique_incidents.items():
                print(f"  Vehicle ID {track_id}: Stopped for {len(frames)} frames "
                      f"(frames {min(frames)}-{max(frames)})")
        
        if output_path:
            print(f"\nAnnotated video saved to: {output_path}")
        
        return all_detections


def main():
    parser = argparse.ArgumentParser(description='Analyze traffic videos for accidents using YOLOv8')
    parser.add_argument('--model', type=str, default='video_detection/yolov8s.pt',
                       help='Path to YOLOv8 model (.pt file)')
    parser.add_argument('--video', type=str, required=True,
                       help='Path to video file to analyze')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save annotated output video')
    parser.add_argument('--preview', action='store_true',
                       help='Show video preview during processing (press q to quit)')
    
    args = parser.parse_args()
    
    # Validate paths
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return
    
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video file not found at {video_path}")
        return
    
    # Setup output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
        # Create output directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Default output in same directory as input
        output_path = video_path.parent / f"{video_path.stem}_analyzed.mp4"
    
    # Run analysis
    detector = AccidentDetector(str(model_path))
    detections = detector.analyze_video(video_path, output_path, args.preview)


if __name__ == "__main__":
    main()