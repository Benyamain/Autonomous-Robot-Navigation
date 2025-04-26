import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import message_filters
import torch
import cv2
import os
import re
import numpy as np

def find_max_frame_index_in_folders(folders, pattern=r'^frame_(\d+)\.'):
    """
    Find the max frame index from folders based on a filename pattern like 'frame_00012.xxx'.
    Return -1 if no matching file is found.
    """
    max_index = -1
    regex = re.compile(pattern)
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for filename in os.listdir(folder):
            match = regex.match(filename)
            if match:
                idx = int(match.group(1))
                if idx > max_index:
                    max_index = idx
    return max_index

class YOLOv5DatasetSaver(Node):
    def __init__(self):
        super().__init__('yolov5_dataset_saver')
        self.bridge = CvBridge()

        # --- setup subscribers (RGB + depth + lidar) ---
        self.rgb_sub = message_filters.Subscriber(self, Image, '/camera_sensor/image_raw')
        self.depth_sub = message_filters.Subscriber(self, Image, '/depth_camera/depth/image_raw')
        self.scan_sub = message_filters.Subscriber(self, LaserScan, '/scan')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.scan_sub],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.sync_callback)

        # --- load model and setup folders ---
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5l6', pretrained=True)
        self.allowed_classes = {0, 28, 56, 60}  # only keep these classes

        self.image_folder = 'dataset_plus/images_2'
        self.annotated_folder = 'dataset_plus/annotated_2'
        self.label_folder = 'dataset_plus/labels_2'
        self.depth_folder = 'dataset_plus/depth_2'
        self.scan_folder = 'dataset_plus/scan_2'

        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.annotated_folder, exist_ok=True)
        os.makedirs(self.label_folder, exist_ok=True)
        os.makedirs(self.depth_folder, exist_ok=True)
        os.makedirs(self.scan_folder, exist_ok=True)

        folders_to_check = [
            self.image_folder,
            self.annotated_folder,
            self.label_folder,
            self.depth_folder,
            self.scan_folder
        ]

        pattern = r'^frame_(\d+)\.'
        existing_max = find_max_frame_index_in_folders(folders_to_check, pattern=pattern)
        if existing_max < 0:
            existing_max = -1
        self.index = existing_max + 1
        self.get_logger().info(f"Found max frame index {existing_max}, new files start from {self.index}")

        self.frame_skip = 1
        self.current_frame = 0

    def sync_callback(self, rgb_msg, depth_msg, scan_msg):
        if self.current_frame % self.frame_skip != 0:
            self.current_frame += 1
            return
        self.current_frame += 1

        # --- convert ros msgs to cv2 images ---
        try:
            cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')
            image = cv_image.copy()
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"Failed to convert images: {str(e)}")
            return

        # --- YOLOv5 detection ---
        results = self.model(cv_image)
        detections = results.xywh[0]

        if detections.numel() == 0:
            annotated_frame = cv_image
            self.get_logger().info("No objects detected, using original image")
        else:
            annotated_frame = results.render()[0]

        cv2.imshow("YOLOv5 Detection (Synced)", annotated_frame)
        cv2.waitKey(1)

        # --- save data if detected classes are allowed ---
        if detections.numel() > 0:
            classes_detected = detections[:, 5].tolist()
            if all(int(cls) in self.allowed_classes for cls in classes_detected):
                # save RGB
                rgb_filename = os.path.join(self.image_folder, f'frame_{self.index:05d}.jpg')
                cv2.imwrite(rgb_filename, image)

                # save annotated image
                ann_filename = os.path.join(self.annotated_folder, f'frame_{self.index:05d}.jpg')
                cv2.imwrite(ann_filename, annotated_frame)

                # save labels
                label_filename = os.path.join(self.label_folder, f'frame_{self.index:05d}.txt')
                with open(label_filename, 'w') as f:
                    for det in detections:
                        x_center, y_center, width, height, conf, cls = det.tolist()
                        f.write(f'{int(cls)} {x_center} {y_center} {width} {height} {conf}\n')

                # save depth
                depth_filename = os.path.join(self.depth_folder, f'frame_{self.index:05d}.npy')
                np.save(depth_filename, depth_image)

                # save lidar
                scan_filename = os.path.join(self.scan_folder, f'frame_{self.index:05d}.npy')
                try:
                    scan_array = np.array(scan_msg.ranges, dtype=np.float32)
                    np.save(scan_filename, scan_array)
                except Exception as e:
                    self.get_logger().error(f"Failed to save scan data: {str(e)}")

                self.get_logger().info(f"Saved data batch {self.index} to {self.image_folder}")

                self.index += 1

    def destroy_node(self):
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YOLOv5DatasetSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
