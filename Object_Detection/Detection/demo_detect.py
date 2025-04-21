# demo_detect.py

from fasterrcnn_infer import load_model, detect_objects, detect_and_plot

# Set paths and parameters
image_path = "./images/test.jpg"  # Replace with your test image path
weights_path = "best_model_distance.pth"
device = "cuda"  # Set to "cpu" if you don't have a GPU

id_to_name = {
    0: 'suitcase',
    1: 'person',
    2: 'table',
    3: 'chair'
}

# Load model
model = load_model(weights_path=weights_path, num_classes=5, device=device)

# Run inference on image
results = detect_and_plot(image_path, model, id_to_name=id_to_name, save_path="result.jpg")

# Print results
print("Detection results:")
for i, det in enumerate(results):
    x, y, cls_id, conf, w, h, dist = det
    print(f"Object {i+1}: Class ID = {cls_id}, Confidence = {conf:.2f}, Distance = {dist:.2f}m, "
          f"Position = ({x:.1f}, {y:.1f}), Size = ({w:.1f}, {h:.1f})")
