import cv2
import time
from fasterrcnn_infer import load_model, detect_objects

# === Parameter Settings ===
input_video = "video.avi"           # Path to the input video
output_video = "output_video.avi"   # Output video with detections
device = "cuda"  # or "cpu"
score_thresh = 0.5

id_to_name = {
    0: 'suitcase',
    1: 'person',
    2: 'table',
    3: 'chair'
}

# === Load Model ===
model = load_model("best_model_distance.pth", num_classes=5, device=device)

# === Open Video ===
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video file: {input_video}")

# Get video info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Create video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"Processing video: {input_video} ({total_frames} frames in total)")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === Start timing for inference
    start = time.time()

    # Save current frame as temporary image
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, frame)

    # Run inference
    results = detect_objects(temp_path, model, device=device, score_thresh=score_thresh)

    # === Draw detection boxes
    for det in results:
        x, y, cls_id, conf, w, h, dist = det
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        label = id_to_name.get(cls_id, f"class {cls_id}")
        label_text = f"{label} | {dist:.2f}m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # === Record inference time
    end = time.time()
    interval = end - start
    print(f"Frame {frame_idx}: Inference time {interval:.3f} seconds")

    # Write frame to output video
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print(f"\nSaved detection result video as {output_video}")
