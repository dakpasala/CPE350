import cv2
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from torchvision import transforms
from ultralytics import YOLO

# ------------------------
# 1. Accident classifier
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf_model = r2plus1d_18(pretrained=True)
clf_model.fc = nn.Linear(clf_model.fc.in_features, 2)
clf_model = clf_model.to(device)
clf_model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                         std=[0.225, 0.225, 0.225])
])

def load_video(path, frames_per_clip=16, step=2):
    cap = cv2.VideoCapture(path)
    frames = []
    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        success, frame = cap.read()
    cap.release()

    if len(frames) < frames_per_clip * step:
        frames += [frames[-1]] * (frames_per_clip * step - len(frames))

    indices = list(range(0, frames_per_clip * step, step))
    clip = [frames[i] for i in indices[:frames_per_clip]]
    clip_tensor = torch.stack([transform(f) for f in clip], dim=1).unsqueeze(0)
    return clip_tensor.to(device), frames

# ------------------------
# 2. Run accident classifier
# ------------------------
video_path = "cr.mp4"
clip_tensor, all_frames = load_video(video_path)

with torch.no_grad():
    outputs = clf_model(clip_tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    accident_prob = probs[1].item()
    pred_label = "ACCIDENT" if accident_prob > 0.5 else "NORMAL"

print(f"Classifier Prediction: {pred_label} (prob={accident_prob:.2f})")

# ------------------------
# 3. Run YOLO detection per frame
# ------------------------
det_model = YOLO("yolov8n.pt")  # can use best.pt once trained

h, w, _ = all_frames[0].shape
out = cv2.VideoWriter("combined_output.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), 6, (w, h))

for frame in all_frames:
    results = det_model.predict(frame, verbose=False)[0]
    annotated = frame.copy()

    # Draw YOLO boxes with class names
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls.item())
        cls_name = det_model.names[cls_id]
        conf = float(box.conf.item())
        color = (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{cls_name} {conf:.2f}",
                    (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2)

    # Add global accident prediction (top-left corner)
    status_color = (0, 0, 255) if pred_label == "ACCIDENT" else (0, 255, 255)
    cv2.putText(annotated, f"Prediction: {pred_label} ({accident_prob:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, status_color, 3)

    out.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

out.release()
print("✅ Combined output saved as combined_output.mp4")