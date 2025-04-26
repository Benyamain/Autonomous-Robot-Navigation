import cv2
import numpy as np
import matplotlib.pyplot as plt

a = "frame_00085"

# --- paths ---
img_path = f"./dataset_1/images_1/{a}.jpg"
label_path = f"./dataset_1/labels_1/{a}.txt"
depth_path = f"./dataset_1/depth_1/{a}.npy"   # depth map
lidar_path = f"./dataset_1/scan_1/{a}.npy"    # lidar scan

image = cv2.imread(img_path)
depth = np.load(depth_path)
lidar = np.load(lidar_path)

img_h, img_w = image.shape[:2]
camera_fov_deg = 60  # camera horizontal FOV (±30°)
lidar_resolution = 2  # lidar has 2° resolution

# --- helper function ---
def x_to_lidar_index(x, image_width, fov_deg):
    angle_in_camera = ((x / image_width) - 0.5) * fov_deg  # -30° to +30°
    angle_lidar = angle_in_camera + 90  # shift to lidar coordinate (60° to 120°)
    idx = int(angle_lidar / lidar_resolution)
    if 0 <= idx < len(lidar):
        return idx
    return None

# --- color map for error ---
def get_color_by_error(error):
    cmap = plt.get_cmap("RdYlGn_r")
    norm_error = min(error / 2.0, 1.0)
    rgb = np.array(cmap(norm_error)[:3]) * 255
    return tuple(map(int, rgb))

# --- read labels and draw comparisons ---
with open(label_path, "r") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    cls, x_center, y_center, box_w, box_h = map(float, line.strip().split())

    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img_w, x2), min(img_h, y2)

    x_center_px = (x1 + x2) / 2
    label = ""
    box_color = (255, 0, 255)  # default color (missing data)

    # --- image depth estimation ---
    region_depth = depth[y1:y2, x1:x2]
    valid_depths = region_depth[(region_depth > 0) & np.isfinite(region_depth)]
    if valid_depths.size > 0:
        image_depth = float(np.median(valid_depths))
        label += f"Img={image_depth:.2f}m"
    else:
        image_depth = None
        label += "Img=N/A"

    # --- lidar matching (allow ±2 indexes) ---
    lidar_idx = x_to_lidar_index(x_center_px, img_w, camera_fov_deg)
    lidar_depth = None
    min_error = float("inf")
    best_match = None
    lidar_tolerance = 2  # ±2 lidar indexes (~±4°)

    if lidar_idx is not None and image_depth is not None:
        start = max(0, lidar_idx - lidar_tolerance)
        end = min(len(lidar), lidar_idx + lidar_tolerance + 1)
        for i in range(start, end):
            if np.isfinite(lidar[i]):
                diff = abs(lidar[i] - image_depth)
                if diff < min_error:
                    min_error = diff
                    best_match = lidar[i]

        if best_match is not None and min_error <= 0.5:  # valid match within 0.5m
            lidar_depth = float(best_match)
            label += f" | LiDAR={lidar_depth:.2f}m"
        else:
            label += " | LiDAR=N/A"

    # --- error calculation ---
    if image_depth is not None and lidar_depth is not None:
        error = abs(image_depth - lidar_depth)
        label += f" | Err={error:.2f}m"
        box_color = get_color_by_error(error)
    else:
        error = None

    # --- draw bbox and label ---
    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
    cv2.putText(image, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, box_color, 2)

# --- show and save ---
cv2.imshow("Depth vs LiDAR Error Map", image)
cv2.imwrite("depth_lidar_error_map_full.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
