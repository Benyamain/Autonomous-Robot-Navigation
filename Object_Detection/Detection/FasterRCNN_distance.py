#!/usr/bin/env python
# coding: utf-8

## Delineate the data set

import os
import random
import shutil

# Set the path
base_dir = os.getcwd()
image_dir = os.path.join(base_dir, 'images_new')
label_dir = os.path.join(base_dir, 'label_distance')

# Destination folder
train_img_dir = os.path.join(base_dir, 'train/images')
train_lbl_dir = os.path.join(base_dir, 'train/labels')
val_img_dir = os.path.join(base_dir, 'val/images')
val_lbl_dir = os.path.join(base_dir, 'val/labels')

# Create folders
for folder in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
    os.makedirs(folder, exist_ok=True)

# Get all image filenames (make sure to press jpg suffix)
all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
random.shuffle(all_images)

# 8:2 Delineation
split_idx = int(len(all_images) * 0.8)
train_files = all_images[:split_idx]
val_files = all_images[split_idx:]

def copy_files(file_list, target_img_dir, target_lbl_dir):
    for img_file in file_list:
        label_file = img_file.replace('.jpg', '.txt')
        src_img = os.path.join(image_dir, img_file)
        src_lbl = os.path.join(label_dir, label_file)
        dst_img = os.path.join(target_img_dir, img_file)
        dst_lbl = os.path.join(target_lbl_dir, label_file)

        shutil.copy2(src_img, dst_img)
        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)

# Copy
copy_files(train_files, train_img_dir, train_lbl_dir)
copy_files(val_files, val_img_dir, val_lbl_dir)

print(f" Data partitioning is complete: training set {len(train_files)} sheets, validation set {len(val_files)} sheets")
# ## Check Dataset

# In[1]:

import os
import cv2
import matplotlib.pyplot as plt

# Configure paths
image_dir = './train/images'
label_dir = './train/labels'

id_to_name = {
    0: 'suitcase',
    1: 'person',
    2: 'table',
    3: 'chair'
}

# Select an image
idx = 19
image_files = sorted(os.listdir(image_dir))
img_name = image_files[idx]
img_path = os.path.join(image_dir, img_name)
label_path = os.path.join(label_dir, img_name.replace('.jpg', '.txt'))

# Read image
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
h, w, _ = image.shape

# Read label and draw
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls_id, cx, cy, bw, bh, dist = parts
            cls_id = int(float(cls_id))

            label = id_to_name.get(cls_id, f"class {cls_id}")
            cx, cy, bw, bh = float(cx)*w, float(cy)*h, float(bw)*w, float(bh)*h
            x1 = int(cx - bw/2)
            y1 = int(cy - bh/2)
            x2 = int(cx + bw/2)
            y2 = int(cy + bh/2)

            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Class + distance label
            label_text = f"{label} | {float(dist):.2f}m"
            cv2.putText(image, label_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

# Display image
plt.figure(figsize=(10, 6))
plt.imshow(image)
plt.title(f"Image: {img_name}")
plt.axis('off')
plt.show()


# In[3]:

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as T
import torchvision
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

#  Dataset class
class YoloToFRCNNDataset(Dataset):
    def __init__(self, image_dir, label_dir, train=True):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.train = train

        # Class mapping: YOLO original ID => sequential index
        self.id_to_name = {
            0: 'suitcase',
            1: 'person',
            2: 'table',
            3: 'chair'
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        boxes = []
        labels = []
        distances = []

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    cls_id, cx, cy, bw, bh, dist = parts
                    cls_id = int(float(cls_id))  # Convert to int

                    label = cls_id

                    # Denormalize
                    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
                    x1 = cx - bw / 2
                    y1 = cy - bh / 2
                    x2 = cx + bw / 2
                    y2 = cy + bh / 2

                    boxes.append([x1, y1, x2, y2])
                    labels.append(label)
                    distances.append(float(dist))

        # Data augmentation
        if self.train:
            if random.random() < 0.5:
                image = F.hflip(image)
                for i in range(len(boxes)):
                    x1, y1, x2, y2 = boxes[i]
                    boxes[i] = [w - x2, y1, w - x1, y2]
            image = F.adjust_brightness(image, 1 + (random.random() - 0.5) * 0.4)
            image = F.adjust_contrast(image, 1 + (random.random() - 0.5) * 0.4)

        image_tensor = T.ToTensor()(image)

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx]),
            "distances": torch.tensor(distances, dtype=torch.float32),
        }

        return image_tensor, target



# In[5]:
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2

def visualize_prediction(model, image_path, device, id_to_name, score_thresh=0.5):
    # Load image
    orig = cv2.imread(image_path)
    image = cv2.cvtColor(orig.copy(), cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Preprocess image
    transform = T.Compose([
        T.ToTensor()
    ])
    img_tensor = transform(image).to(device)

    # Model inference
    model.eval()
    with torch.no_grad():
        outputs = model([img_tensor])[0]

    boxes = outputs['boxes'].cpu()
    labels = outputs['labels'].cpu()
    scores = outputs['scores'].cpu()
    distances = outputs['distances'].cpu() if 'distances' in outputs else None

    # Draw boxes
    for box, label, score, dist in zip(boxes, labels, scores, distances):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = map(int, box)
        name = id_to_name.get(label.item(), f"cls {label.item()}")
        dist_str = f"{dist:.2f}m" if distances is not None else ""
        label_str = f"{name} | {dist_str}"

        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig, label_str, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Display image
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {image_path}")
    plt.axis('off')
    plt.show()


# In[9]:


#from model_with_distance import FasterRCNNWithDistance
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

# Category mapping (your continuous ID)
id_to_name = {
    0: 'suitcase',
    1: 'person',
    2: 'table',
    3: 'chair'
}

# Model loading
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
backbone = resnet_fpn_backbone('resnet101', pretrained=False)
model = FasterRCNNWithDistance(backbone, num_classes=5).to(device)
model.load_state_dict(torch.load("best_model_distance.pth", map_location=device))
model.eval()

# demo a image
image_path = "./val/images/frame_00000_2.jpg"  
visualize_prediction(model, image_path, device, id_to_name)


# In[11]:


def collate_fn(batch):
    return tuple(zip(*batch))
#  Model Construction (ResNet101 + FPN)
def get_model(num_classes):
    backbone = resnet_fpn_backbone('resnet101', pretrained=True)
    model = FasterRCNNWithDistance(backbone, num_classes)
    return model


#  Evaluation Function (Per-Epoch Evaluation)
import torch
from torchvision.ops import box_iou

@torch.no_grad()
def evaluate_model(model, val_loader, device="cuda", iou_threshold=0.6, score_threshold=0.5):
    model.eval()
    model.to(device)

    total_preds = 0
    total_gts = 0
    correct = 0
    correct_total_images = 0
    distance_errors = []  #  Record the difference between predicted and GT distances

    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(images)

        for pred, target in zip(outputs, targets):
            pred_boxes = pred['boxes']
            pred_scores = pred['scores']
            pred_labels = pred['labels']
            pred_dists = pred['distances'] if 'distances' in pred else None

            gt_boxes = target['boxes']
            gt_labels = target['labels']
            gt_dists = target['distances']

            keep = pred_scores > score_threshold
            pred_boxes = pred_boxes[keep]
            pred_labels = pred_labels[keep]
            if pred_dists is not None:
                keep = keep.to(pred_dists.device)  #  Ensure `keep` and `pred_dists` are on the same device
                pred_dists = pred_dists[keep]

            total_preds += len(pred_boxes)
            total_gts += len(gt_boxes)

            matched = 0
            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                ious = box_iou(pred_boxes, gt_boxes)
                for i in range(len(pred_boxes)):
                    max_iou, gt_idx = ious[i].max(0)
                    if max_iou > iou_threshold and pred_labels[i] == gt_labels[gt_idx]:
                        correct += 1
                        matched += 1
                        ious[:, gt_idx] = 0  # Prevent duplicate matching

                        #  Calculate distance error
                        if pred_dists is not None and gt_idx < len(gt_dists):
                            pred_dist = pred_dists[i].item()
                            gt_dist = gt_dists[gt_idx].item()
                            distance_errors.append(abs(pred_dist - gt_dist))

            if matched > 0:
                correct_total_images += 1

    precision = correct / total_preds if total_preds > 0 else 0
    recall = correct / total_gts if total_gts > 0 else 0
    accuracy = correct_total_images / len(val_loader.dataset)

    avg_distance_error = sum(distance_errors) / len(distance_errors) if distance_errors else -1

    return precision, recall, accuracy, avg_distance_error


#  Data & Model Preparation
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = YoloToFRCNNDataset('./train/images', './train/labels', train=True)
val_dataset = YoloToFRCNNDataset('./val/images', './val/labels', train=False)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
model = get_model(num_classes=5)
model.to(device)
print(torch.cuda.is_available())  # True means GPU is available
print(device)                     # Should print 'cuda'


# load best_loss
best_loss = float('inf')
# if os.path.exists("best_model_resnet101.pth"):
#     model.load_state_dict(torch.load("best_model_resnet101.pth", map_location=device))
#     print(" Loaded best_model_resnet101.pth")

if os.path.exists("best_loss.txt"):
    with open("best_loss.txt", "r") as f:
        best_loss = float(f.read())
    print(f" Loaded previous best loss: {best_loss:.4f}")

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

# Training + Evaluation Master Cycle
num_epochs = 50
loss_list = []
precision_list = []
recall_list = []
acc_list = []

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f" Epoch {epoch+1}/{num_epochs}")

    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    
        # Filter out samples with 0 boxes
        filtered_images = []
        filtered_targets = []
        for img, tgt in zip(images, targets):
            if tgt['boxes'].numel() > 0:
                filtered_images.append(img)
                filtered_targets.append(tgt)
    
        if len(filtered_images) == 0:
            continue  # All empty labels, skip the batch
    
        loss_dict = model(filtered_images, filtered_targets)
        #print(loss_dict)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        total_loss += losses.item()
        pbar.set_postfix({k: f"{v.item():.4f}" for k, v in loss_dict.items()})

    avg_loss = total_loss / len(train_loader)
    loss_list.append(avg_loss)
    lr_scheduler.step(avg_loss)

    # Evaluated in every round
    precision, recall, acc, dist_err = evaluate_model(model, val_loader, device=device)

    print(f"\n Epoch {epoch+1}: Avg Loss={avg_loss:.4f} |  Precision={precision:.4f} |  Recall={recall:.4f} |  Accuracy={acc:.4f} |  AvgDistError={dist_err:.2f}m\n")

    precision_list.append(precision)
    recall_list.append(recall)
    acc_list.append(acc)

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "best_model_distance.pth")
        with open("best_loss.txt", "w") as f:
            f.write(str(best_loss))
        print(" Saved best model.")

# Training completed, plotting multiple comparisons
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(loss_list, marker='o')
plt.title("Loss")
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(precision_list, marker='o', label='Precision')
plt.plot(recall_list, marker='s', label='Recall')
plt.title("Precision / Recall")
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(acc_list, marker='^', color='green')
plt.title("Accuracy (image level)")
plt.grid(True)

plt.tight_layout()
plt.show()


# In[13]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[15]:


print(device)


# In[ ]:




