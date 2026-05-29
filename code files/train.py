# DBMANet training + evaluation script for WHU LongKou dataset
# Set dataset_name = 'WHU-Hi-LongKou' to run on LongKou

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, classification_report
import random
import shutil

# -----------------------
# Config
# -----------------------
dataset_root = '/kaggle/input/whu-hyperspectral-dataset'
dataset_name = 'WHU-Hi-LongKou'   # <-- LongKou
work_dir = '/kaggle/working/dataset'
os.makedirs(work_dir, exist_ok=True)

# Training params
patch_size = 5
batch_size = 32
num_epochs = 50
learning_rate = 1e-3
test_ratio = 0.9   # fraction used for training (rest for testing)
num_workers = 2
pin_memory = True

# Reproducibility & device
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.empty_cache()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------
# Prepare dataset files
# -----------------------
src_dir = os.path.join(dataset_root, dataset_name)
if not os.path.isdir(src_dir):
    raise FileNotFoundError(f"Dataset folder not found: {src_dir}")

dst_dir = os.path.join(work_dir, dataset_name.split('/')[-1])
os.makedirs(dst_dir, exist_ok=True)

# find .mat files (one data .mat and one gt .mat)
files = os.listdir(src_dir)
data_file = None
gt_file = None
for f in files:
    if f.endswith('.mat') and ('gt' not in f.lower()):
        data_file = f
    if f.endswith('.mat') and ('gt' in f.lower()):
        gt_file = f
if data_file is None or gt_file is None:
    raise FileNotFoundError(f"Couldn't find data/gt .mat files in {src_dir}. Found: {files}")

shutil.copy(os.path.join(src_dir, data_file), os.path.join(dst_dir, 'data.mat'))
shutil.copy(os.path.join(src_dir, gt_file), os.path.join(dst_dir, 'gt.mat'))
print("Copied dataset files:", data_file, gt_file)

# -----------------------
# Load data
# -----------------------
data_mat = sio.loadmat(os.path.join(dst_dir, 'data.mat'))
gt_mat = sio.loadmat(os.path.join(dst_dir, 'gt.mat'))

def find_array(matdict, ndim=None):
    for k, v in matdict.items():
        if k.startswith('__'):
            continue
        if isinstance(v, np.ndarray):
            if ndim is None or v.ndim == ndim:
                return v
    raise ValueError("No ndarray found in mat file.")

# prefer 3D array for data, 2D for gt
data_arr = None
gt_arr = None
for k, v in data_mat.items():
    if isinstance(v, np.ndarray) and v.ndim == 3:
        data_arr = v
        break
if data_arr is None:
    data_arr = find_array(data_mat)

for k, v in gt_mat.items():
    if isinstance(v, np.ndarray) and v.ndim == 2:
        gt_arr = v
        break
if gt_arr is None:
    gt_arr = find_array(gt_mat)

data = data_arr.astype(np.float32)
gt = gt_arr.astype(np.int32)
print("Data shape:", data.shape, "GT shape:", gt.shape)
unique_labels = np.unique(gt)
print("Unique labels (incl 0):", unique_labels)
num_classes = int(np.max(gt))
print("Number of classes:", num_classes)

# normalize data to [0,1]
data = (data - data.min()) / (data.max() - data.min() + 1e-12)

# -----------------------
# Dataset class
# -----------------------
class HyperSpectralDataset(Dataset):
    def __init__(self, data, gt, patch_size=5, train=True, split_ratio=0.9):
        self.data = data
        self.gt = gt
        self.patch_size = patch_size
        self.pad = patch_size // 2
        self.padded = np.pad(data, ((self.pad,self.pad),(self.pad,self.pad),(0,0)), mode='reflect')
        indices = np.argwhere(gt > 0)
        np.random.seed(42)
        np.random.shuffle(indices)
        split = int(len(indices) * split_ratio)
        self.indices = indices[:split] if train else indices[split:]
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        x, y = self.indices[idx]
        xp = x + self.pad; yp = y + self.pad
        patch = self.padded[xp-self.pad:xp+self.pad+1, yp-self.pad:yp+self.pad+1, :]
        patch = torch.tensor(patch, dtype=torch.float32).permute(2,0,1)
        label = int(self.gt[x,y]) - 1
        return patch, torch.tensor(label, dtype=torch.long)

# -----------------------
# Model (DBMANet)
# -----------------------
class SpectralBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class SpatialBranch(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class AttentionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(channels*2, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
    def forward(self, f1, f2):
        fused = torch.cat([f1,f2], dim=1)
        a = self.att(fused)
        return a * f1 + (1 - a) * f2

class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )
    def forward(self,x): return self.head(x)

class DBMANet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.spec = SpectralBranch(in_channels)
        self.spat = SpatialBranch(in_channels)
        self.fuse = AttentionFusion(64)
        self.cls = ClassificationHead(64, num_classes)
    def forward(self, x):
        f1 = self.spec(x)
        f2 = self.spat(x)
        fused = self.fuse(f1,f2)
        return self.cls(fused)

# -----------------------
# Datasets & loaders
# -----------------------
train_dataset = HyperSpectralDataset(data, gt, patch_size=patch_size, train=True, split_ratio=test_ratio)
test_dataset  = HyperSpectralDataset(data, gt, patch_size=patch_size, train=False, split_ratio=test_ratio)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
test_loader  = DataLoader(test_dataset,  batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

# -----------------------
# Setup model, loss, optimizer
# -----------------------
model = DBMANet(in_channels=data.shape[2], num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# -----------------------
# Training loop
# -----------------------
for epoch in range(1, num_epochs+1):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    for xb, yb in train_loader:
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        running_correct += (preds == yb).sum().item()
        running_total += xb.size(0)
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    print(f"Epoch {epoch:03d}  Loss: {epoch_loss:.4f}  Train Acc: {epoch_acc*100:.2f}%")

# -----------------------
# Evaluation
# -----------------------
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device, dtype=torch.float32)
        out = model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(yb.numpy().tolist())

all_preds = np.array(all_preds, dtype=np.int64)
all_labels = np.array(all_labels, dtype=np.int64)

cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_classes)))
oa = accuracy_score(all_labels, all_preds)
true_counts = cm.sum(axis=1)
per_class_acc = np.zeros(num_classes, dtype=np.float32)
for i in range(num_classes):
    per_class_acc[i] = (cm[i,i] / true_counts[i]) if true_counts[i] > 0 else 0.0
kappa = cohen_kappa_score(all_labels, all_preds)

print("\n=== Final Evaluation ===")
print(f"Dataset: {dataset_name}")
print(f"Overall Accuracy (OA): {oa*100:.2f}%")
for i, acc in enumerate(per_class_acc, start=1):
    print(f"Class {i} Accuracy: {acc*100:.2f}%  (N={int(true_counts[i-1])})")
print(f"\nCohen's Kappa: {kappa:.4f}")
print("\nConfusion Matrix (rows=true classes 1..K; cols=predicted 1..K):")
print(cm)
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

# save model
torch.save(model.state_dict(), f"dbmanet_{dataset_name}.pth")
print("Saved model:", f"dbmanet_{dataset_name}.pth")
hey add average accuracy (AA) in it are you getting