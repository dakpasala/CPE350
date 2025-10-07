import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models.video import r2plus1d_18
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import os
import cv2

# ------------------------
# 1. Define video dataset
# ------------------------
class VideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames_per_clip=16, step=2):
        """
        root_dir: path to dataset split (e.g. data/train)
        transform: torchvision transforms
        frames_per_clip: number of frames in each video clip
        step: frame sampling step
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.step = step
        self.samples = []

        # Collect video file paths and labels
        for label, cls in enumerate(["normal", "accident"]):
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.endswith((".mp4", ".avi", ".mov")):
                    self.samples.append((os.path.join(cls_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        # Load video using OpenCV
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        while success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert to RGB
            frames.append(frame)
            success, frame = cap.read()
        cap.release()

        # Sample frames (uniform step)
        if len(frames) < self.frames_per_clip * self.step:
            # pad by repeating last frame
            frames += [frames[-1]] * (self.frames_per_clip * self.step - len(frames))

        indices = list(range(0, self.frames_per_clip * self.step, self.step))
        clip = [frames[i] for i in indices[:self.frames_per_clip]]

        # Apply transforms frame-by-frame
        if self.transform:
            clip = [self.transform(frame) for frame in clip]

        # Shape: (C, T, H, W)
        clip = torch.stack(clip, dim=1)
        return clip, label


# ------------------------
# 2. Define transforms
# ------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),   # smaller = faster
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45],
                         std=[0.225, 0.225, 0.225])
])

# ------------------------
# 3. Create datasets + loaders
# ------------------------
train_dataset = VideoDataset("data/train", transform=transform)
val_dataset = VideoDataset("data/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

# ------------------------
# 4. Load pretrained model
# ------------------------
model = r2plus1d_18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)  # binary classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------------
# 5. Training setup
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ------------------------
# 6. Training loop (skeleton)
# ------------------------
for epoch in range(5):  # small number for testing
    model.train()
    for clips, labels in train_loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

print("✅ Training loop finished (starter version)")
