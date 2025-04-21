
import torch
import torchvision.transforms as T
from PIL import Image
import cv2
import numpy as np
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
import torch.nn as nn
import matplotlib.pyplot as plt

# === Our FasterRCNN Model, adding distance branch ===
class FasterRCNNWithDistance(FasterRCNN):
    def __init__(self, backbone, num_classes):
        super().__init__(backbone, num_classes)
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.distance_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, images, targets=None):
        if self.training:
            loss_dict = super().forward(images, targets)
            features = self.backbone(images)
            all_gt_boxes = [t["boxes"] for t in targets]
            image_shapes = [img.shape[1:] for img in images]
            box_features = self.roi_heads.box_roi_pool(features, all_gt_boxes, image_shapes)
            box_features = self.roi_heads.box_head(box_features)
            pred_distances = self.distance_head(box_features).squeeze(1)
            gt_distances = torch.cat([t["distances"] for t in targets]).to(pred_distances.device)
            distance_loss = nn.functional.smooth_l1_loss(pred_distances, gt_distances)
            loss_dict["loss_distance"] = distance_loss
            return loss_dict
        else:
            detections = super().forward(images)
            features = self.backbone(images)
            all_boxes = [d["boxes"] for d in detections]
            image_shapes = [img.shape[1:] for img in images]
            if sum(len(b) for b in all_boxes) == 0:
                for det in detections:
                    det["distances"] = torch.tensor([]).to(images[0].device)
                return detections
            box_features = self.roi_heads.box_roi_pool(features, all_boxes, image_shapes)
            box_features = self.roi_heads.box_head(box_features)
            pred_distances = self.distance_head(box_features).squeeze(1)
            start = 0
            for i in range(len(detections)):
                num_boxes = len(detections[i]["boxes"])
                if num_boxes == 0:
                    detections[i]["distances"] = torch.tensor([]).to(images[0].device)
                else:
                    detections[i]["distances"] = pred_distances[start:start+num_boxes].detach().cpu()
                    start += num_boxes
            return detections

# === load the model ===
def load_model(weights_path="best_model_distance.pth", num_classes=5, device="cuda"):
    backbone = resnet_fpn_backbone('resnet101', pretrained=False)
    model = FasterRCNNWithDistance(backbone, num_classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# === Inference function, returns a list of results ===
# boxes, labels, scores, distances
def detect_objects(image_path, model, device="cuda", score_thresh=0.5):
    image = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    distances = outputs["distances"].cpu().numpy() if "distances" in outputs else np.zeros(len(boxes))

    results = []
    for i in range(len(boxes)):
        if scores[i] < score_thresh:
            continue
        x1, y1, x2, y2 = boxes[i]
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        results.append([x_center, y_center, labels[i], scores[i], width, height, distances[i]])

    return results

# === Reasoning + display target frame image ===
def detect_and_plot(image_path, model, id_to_name=None, device="cuda", score_thresh=0.5, save_path="result.jpg"):
    image = cv2.imread(image_path)
    orig = image.copy()
    image_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    transform = T.ToTensor()
    img_tensor = transform(image_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()
    distances = outputs["distances"].cpu().numpy()

    for i in range(len(boxes)):
        if scores[i] < score_thresh:
            continue
        x1, y1, x2, y2 = boxes[i].astype(int)
        dist = distances[i]
        label = labels[i]
        name = id_to_name[label] if id_to_name and label in id_to_name else f"class {label}"
        text = f"{name} | {dist:.2f}m"

        cv2.rectangle(orig, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(orig, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # save the image
    cv2.imwrite(save_path, orig)
    print(f" detection results have been saved in {save_path}")

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
    plt.title("Detection Results")
    plt.axis("off")
    plt.show()
