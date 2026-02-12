"""
Accident Detection Script using YOLO
Analyzes traffic camera footage for accidents using a trained YOLO model
"""

import os

# CRITICAL: Fix for PyTorch 2.6 compatibility - must be before other imports
os.environ['TORCH_FORCE_WEIGHTS_ONLY_LOAD'] = '0'

import torch
_original_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _original_load(*args, **kwargs)
torch.load = _patched_load

import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import argparse
from pathlib import Path


class AccidentDetector:
    def __init__(self, model_path, conf_threshold=0.4):
        """
        Initialize the accident detector with YOLO model
        
        Args:
            model_path: Path to the YOLO .pt model file (e.g., best.pt)
            conf_threshold: Confidence threshold for detections (default: 0.4)
        """
        print(f"Loading model from: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.accident_detections = []
        
    def analyze_frame(self, frame, frame_number):
        """
        Analyze a single frame for accidents
        
        Args:
            frame: Video frame to analyze
            frame_number: Current frame number
            
        Returns:
            annotated_frame: Frame with bounding boxes
            detections: List of accidents detected in this frame
        """
        # Run YOLO detection with confidence threshold
        results = self.model.predict(
            source=frame, 
            conf=self.conf_threshold,
            verbose=False
        )
        
        annotated_frame = frame.copy()
        frame_detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            # Get class names from the model
            class_names = results[0].names
            
            for box, cls, conf in zip(boxes, classes, confidences):
                x1, y1, x2, y2 = box.astype(int)
                class_name = class_names[cls]
                
                # Record detection
                frame_detections.append({
                    'frame': frame_number,
                    'class': class_name,
                    'confidence': float(conf),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                
                # Draw bounding box (red for accidents)
                color = (0, 0, 255)  # Red for accidents
                thickness = 3
                
                cv2.rectangle(annotated_frame, 
                            (x1, y1), 
                            (x2, y2), 
                            color, thickness)
                
                # Add label with class name and confidence
                label = f"{class_name}: {conf:.2f}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw label background
                cv2.rectangle(annotated_frame,
                            (x1, y1 - label_size[1] - 10),
                            (x1 + label_size[0], y1),
                            color, -1)
                
                # Draw label text
                cv2.putText(annotated_frame, label,
                          (x1, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame, frame_detections
    
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
        
        print(f"\n{'='*60}")
        print(f"Video: {video_path.name}")
        print(f"Resolution: {width}x{height}")
        print(f"FPS: {fps}")
        print(f"Total Frames: {total_frames}")
        print(f"Duration: {total_frames/fps:.2f} seconds")
        print(f"{'='*60}\n")
        
        # Setup video writer if output path specified
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_number = 0
        all_detections = []
        
        print("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            annotated_frame, detections = self.analyze_frame(frame, frame_number)
            all_detections.extend(detections)
            
            # Add frame counter and detection count
            info_text = f"Frame: {frame_number}/{total_frames}"
            cv2.putText(annotated_frame, info_text,
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            if detections:
                detection_text = f"ACCIDENTS DETECTED: {len(detections)}"
                cv2.putText(annotated_frame, detection_text,
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Write output
            if out:
                out.write(annotated_frame)
            
            # Show preview
            if show_preview:
                cv2.imshow('Accident Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nStopped by user")
                    break
            
            frame_number += 1
            
            # Progress indicator
            if frame_number % 30 == 0 or frame_number == total_frames:
                percent = (frame_number / total_frames) * 100
                print(f"Progress: {frame_number}/{total_frames} frames ({percent:.1f}%)", end='\r')
        
        cap.release()
        if out:
            out.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        # Print summary
        print(f"\n\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        print(f"Total frames processed: {frame_number}")
        print(f"Total detections: {len(all_detections)}")
        
        if all_detections:
            # Count detections by class
            class_counts = defaultdict(int)
            for det in all_detections:
                class_counts[det['class']] += 1
            
            print(f"\nDetection Summary:")
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count} detections")
            
            # Show frames with detections
            frames_with_detections = set(det['frame'] for det in all_detections)
            print(f"\nFrames with accidents: {len(frames_with_detections)}")
            
            # Show first few detections
            print(f"\nFirst 10 detections:")
            for i, det in enumerate(all_detections[:10]):
                print(f"  Frame {det['frame']}: {det['class']} (confidence: {det['confidence']:.2f})")
            
            if len(all_detections) > 10:
                print(f"  ... and {len(all_detections) - 10} more")
        else:
            print("\nNo accidents detected in this video.")
        
        if output_path:
            print(f"\n{'='*60}")
            print(f"Annotated video saved to: {output_path}")
            print(f"{'='*60}")
        
        return all_detections
    
    def analyze_image(self, image_path, output_path=None):
        """
        Analyze a single image for accidents
        
        Args:
            image_path: Path to input image
            output_path: Optional path to save annotated image
        """
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not read image {image_path}")
            return
        
        print(f"\nAnalyzing image: {image_path.name}")
        
        # Analyze the image
        annotated_image, detections = self.analyze_frame(image, 0)
        
        # Print results
        if detections:
            print(f"✓ ACCIDENTS DETECTED: {len(detections)}")
            for det in detections:
                print(f"  - {det['class']} (confidence: {det['confidence']:.2f})")
        else:
            print("✓ No accidents detected.")
        
        # Save output
        if output_path:
            cv2.imwrite(str(output_path), annotated_image)
            print(f"\nAnnotated image saved to: {output_path}")
        
        return detections


def main():
    parser = argparse.ArgumentParser(
        description='Analyze traffic videos/images for accidents using YOLO',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze a video
  python analyze_video.py --video videos/youtube.mp4
  
  # Analyze with preview window
  python analyze_video.py --video videos/youtube.mp4 --preview
  
  # Analyze an image
  python analyze_video.py --image test.jpg
  
  # Adjust confidence threshold
  python analyze_video.py --video videos/youtube.mp4 --conf 0.5
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str,
                       help='Path to video file to analyze')
    group.add_argument('--image', type=str,
                       help='Path to image file to analyze')
    
    parser.add_argument('--preview', action='store_true',
                       help='Show preview during processing (video only, press q to quit)')
    parser.add_argument('--conf', type=float, default=0.4,
                       help='Confidence threshold for detections (default: 0.4)')
    
    args = parser.parse_args()
    
    # Hardcoded model path
    model_path = Path("video_detection/yolov8s.pt")
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Make sure video_detection/yolov8s.pt exists in your project directory")
        return
    
    # Initialize detector
    detector = AccidentDetector(str(model_path), conf_threshold=args.conf)
    
    # Analyze video or image
    if args.video:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found at {video_path}")
            return
        
        # Hardcoded output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"analyzed_{video_path.name}"
        
        # Analyze video
        detector.analyze_video(video_path, output_path, args.preview)
        
    else:  # args.image
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found at {image_path}")
            return
        
        # Hardcoded output directory
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"analyzed_{image_path.name}"
        
        # Analyze image
        detector.analyze_image(image_path, output_path)


if __name__ == "__main__":
    main()