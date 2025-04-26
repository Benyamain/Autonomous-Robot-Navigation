import cv2
import os

# set input and output folders
input_folder = "/home/admin1/yolov5/dataset/images"  # change this to your image folder
output_folder = "/home/admin1/yolov5/dataset/Rotate_Expand"  # folder to save rotated images

# make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# get all image files (jpg, png, jpeg)
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# process each image
for image_file in image_files:
    img_path = os.path.join(input_folder, image_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to read {image_file}, skipping...")
        continue

    # rotate the image (90 degrees clockwise)
    rotated_img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    # save the rotated image
    output_path = os.path.join(output_folder, f"rotated_{image_file}")
    cv2.imwrite(output_path, rotated_img)

    print(f"Done: {image_file} -> {output_path}")

print("All images processed!")
