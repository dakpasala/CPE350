import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
from torchvision import transforms
import cv2

# ------------------------
# 1. Load model (binary classifier)
# ------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = r2plus1d_18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # accident vs normal
model = model.to(device)

# Load trained weights (if you fine-tuned already)
# model.load_state_dict(torch.load("accident_model.pth"))

model.eval()

# ------------------------
# 2. Define transforms
# ------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                         std=[0.225, 0.225, 0.225])
])

# ------------------------
# 3. Helper: Load video into clip tensor
# ------------------------
def load_video(path, frames_per_clip=16, step=2):
    cap = cv2.VideoCapture(path)
    frames = []
    success, frame = cap.read()
    while success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        success, frame = cap.read()
    cap.release()

    # Pad if too short
    if len(frames) < frames_per_clip * step:
        frames += [frames[-1]] * (frames_per_clip * step - len(frames))

    # Uniform sampling
    indices = list(range(0, frames_per_clip * step, step))
    clip = [frames[i] for i in indices[:frames_per_clip]]

    clip = [transform(frame) for frame in clip]
    clip = torch.stack(clip, dim=1)  # (C, T, H, W)
    return clip.unsqueeze(0)         # add batch dim → (1, C, T, H, W)

# ------------------------
# 4. Run inference
# ------------------------
video_path = "cr.mp4"
clip = load_video(video_path).to(device)

with torch.no_grad():
    outputs = model(clip)
    probs = torch.softmax(outputs, dim=1)[0]
    pred_class = torch.argmax(probs).item()

labels = ["normal", "accident"]
print(f"Prediction: {labels[pred_class]} (prob={probs[pred_class]:.2f})")
