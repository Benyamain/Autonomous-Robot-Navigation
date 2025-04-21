After I am training the FasterRCNN with distance, I got the model, and then I can use this model file to detect a image or a video to check the result

This part implements an extended Faster R-CNN object detection model that not only detects bounding boxes and class labels 
but also predicts the distance to each object using a custom regression head. 
The framework is designed to support real-time image and video inference.

- ├── FasterRCNN_distance.py       # Main training + evaluation + visualization script
- ├── fasterrcnn_infer.py          # Inference model class + utility functions
- ├── demo_detect.py               # Demo: run inference on a single image
- ├── demo_video.py                # Demo: run inference on a video file
- ├── best_model_distance.pth      # (optional) Trained model checkpoint
- |-------Link: https://udmercy0-my.sharepoint.com/:u:/g/personal/zhangxi24_udmercy_edu/Ec2KFie_a_5Jm0s5-asdqb8Bc4Hotjqb8a-dipG6K0OdQw?e=fwLpok
- ├── images_new/                  # Raw input images
- ├── label_distance/              # YOLO-format labels with distance (x, y, w, h, dist)
- ├── train/val/                   # Auto-generated split datasets
- └── result.jpg                   # Sample output image (after detection)

