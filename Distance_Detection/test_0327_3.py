import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import matplotlib.patches as patches
from collections import defaultdict

depth_map = np.load('C:/Users/wangji15/Downloads/depth_/depth_/depth/frame_02775.npy')
# depth_map = np.load('C:/Users/wangji15/Downloads/depth_/depth_/depth/frame_00029.npy')
# 相机内参 + 俯角
fy   = 1108.7654256452881
cy   = 360.5
tilt = 0.2  

H, W = depth_map.shape

row_max = np.max(depth_map, axis=1)
diffs = np.diff(row_max)
threshold_jump = 0.1
row_max_filtered = row_max.copy()

for i in range(len(diffs)):
    if abs(diffs[i]) > threshold_jump:
        row_max_filtered[i+1] = 0.0  
#这个值过大可提升滤除劈尖效果，但同时也将失去物体的细节（导致物体变细）
tolerance = 0.25  # 最大容差(Best = 0.25~0.35)

depth_filtered = depth_map.copy()

for row in range(H):
    max_val = row_max_filtered[row]
    if max_val <= 0: 
        continue
    diff_row = np.abs(depth_filtered[row, :] - max_val)
    mask = diff_row < tolerance

    depth_filtered[row, mask] = 0.0  

# 每行像素对应与光轴的垂直夹角(考虑俯角)
v_coords = np.arange(H)
theta_v = np.arctan((v_coords - cy) / fy)
theta_v_adjusted = theta_v - tilt

cos_theta = np.cos(theta_v_adjusted).reshape(H, 1)

depth_z = depth_filtered * cos_theta
# 这个值决定在区域生长中，不同区域之间的突变轮廓。
DEPTH_DIFF_TH = 0.05  # 突变阈值(Best = 0.05~0.1)

def get_neighbors_8(h, w, x, y):
    """返回8邻域坐标"""
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w:
                neighbors.append((nx, ny))
    return neighbors

def region_grow_segmentation(depth_img, diff_thresh=0.1):
    """
    在给定深度图上做区域生长。
    diff_thresh = 0.1 (m)，当两个相邻像素深度之差 < 0.1时视为同一物体。
    depth_img: (H, W), 这里使用修正后的 Z-depth。
    返回：label_map，同尺寸 (H,W)，区域ID从1开始递增。
         region_count：最终划分出的区域数目（不含无效点）。
    """
    h, w = depth_img.shape
    label_map = np.zeros((h, w), dtype=np.int32)
    current_label = 0

    for i in range(h):
        for j in range(w):
            if label_map[i, j] != 0:
                continue

            base_depth = depth_img[i, j]
            if base_depth <= 0 or np.isnan(base_depth):
                label_map[i, j] = -1
                continue

            current_label += 1
            label_map[i, j] = current_label
            
            # BFS队列
            queue = deque()
            queue.append((i, j))

            while queue:
                x, y = queue.popleft()
                depth_xy = depth_img[x, y]

                # 遍历8邻域
                for nx, ny in get_neighbors_8(h, w, x, y):
                    if label_map[nx, ny] != 0:
                        continue

                    neighbor_depth = depth_img[nx, ny]
                    # 若有效 且 深度差<阈值 -> 同一物体
                    if neighbor_depth > 0 and not np.isnan(neighbor_depth):
                        if abs(neighbor_depth - depth_xy) < diff_thresh:
                            label_map[nx, ny] = current_label
                            queue.append((nx, ny))

    return label_map, current_label

# 调用区域生长
label_map, region_count = region_grow_segmentation(depth_z, diff_thresh=DEPTH_DIFF_TH)

label_viz = np.zeros((H, W, 3), dtype=np.uint8)
max_label = label_map.max()

for i in range(H):
    for j in range(W):
        lbl = label_map[i, j]
        if lbl <= 0:  # -1 或 0
            label_viz[i, j] = (0, 0, 0)  # 黑
        else:
            color_r = (lbl * 37) % 255
            color_g = (lbl * 73) % 255
            color_b = (lbl * 109) % 255
            label_viz[i, j] = (color_r, color_g, color_b)

plt.subplot(1, 1, 1)
plt.imshow(label_viz)
plt.title(f"Region-Growing (Threshold={DEPTH_DIFF_TH}, {region_count} regions)")
plt.axis('off')

plt.tight_layout()
plt.show()

print(f"Done. Total {region_count} regions (excluding invalid).")

# -------------------------------
# Step 1: 相机投影辅助函数
# -------------------------------

def pixel_to_cam(u, v, d, fx, fy, cx, cy):
    X = (u - cx) * d / fx
    Y = (v - cy) * d / fy
    Z = d
    return np.array([X, Y, Z, 1.0], dtype=np.float32)

def transform_to_world(pt_cam, T_world_cam):
    return T_world_cam @ pt_cam

def flatten_to_ground(pt_world):
    Xw, Yw, _, _ = pt_world  # 直接忽略 Z
    return (Xw, Yw)

fx = 1108.77
fy = 1108.77 
cx = 640
cy = 360

Ry = np.array([
    [ 1,             0,            0,        0],
    [ 0,  -0.198669324, -0.980066579, 1.960133],
    [ 0,   0.980066579, -0.198669324, 0.397339],
    [ 0,             0,            0,        1]
])
T_world_cam = Ry
#cam_origin = (Ry @ np.array([0, 0, 0, 1]))[:3]
#print(cam_origin)

region_ground_points = defaultdict(list)

for i in range(H):
    for j in range(W):
        region_id = label_map[i, j]
        if region_id <= 0:
            continue

        depth_val = depth_filtered[i, j]
        if depth_val <= 0 or np.isnan(depth_val):
            continue

        u, v = j, i
        pt_cam = pixel_to_cam(u, v, depth_val, fx, fy, cx, cy)
        pt_world = transform_to_world(pt_cam, T_world_cam)
        xg, yg = flatten_to_ground(pt_world)
        region_ground_points[region_id].append((xg, yg))

plt.figure(figsize=(10, 8))
for region_id, points in region_ground_points.items():
    if len(points) < 5:
        continue
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    color = (region_id * 37 % 255 / 255,
             region_id * 73 % 255 / 255,
             region_id * 109 % 255 / 255)
    plt.scatter(xs, ys, s=5, c=[color], label=f'Region {region_id}')

plt.title("Ground Plane Projection of Regions")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 8))

for region_id, points in region_ground_points.items():
    if len(points) < 5:
        continue  # 太小的区域忽略

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # 画点
    color = (region_id * 37 % 255 / 255,
             region_id * 73 % 255 / 255,
             region_id * 109 % 255 / 255)
    plt.scatter(xs, ys, s=5, c=[color], label=f'Region {region_id}')

    # === 外接矩形计算 ===
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    width = xmax - xmin
    height = ymax - ymin

    # 添加矩形框
    rect = patches.Rectangle((xmin, ymin), width, height,
                             linewidth=1.5, edgecolor='red', facecolor='none')
    plt.gca().add_patch(rect)

plt.title("Ground Plane Projection with Bounding Boxes")
plt.xlabel("X (meters)")
plt.ylabel("Y (meters)")
plt.grid(True)
plt.axis("equal")
plt.tight_layout()
plt.show()
