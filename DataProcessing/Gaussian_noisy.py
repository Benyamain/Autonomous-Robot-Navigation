import cv2
import numpy as np
import os

# set input and output folders
input_folder = "/home/admin1/yolov5/dataset/images"  # your image folder path
output_folder = "/home/admin1/yolov5/dataset/Gaussian_Noisy"  # path to save noisy images

# make sure output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# get all image files (jpg and png)
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png'))]

# gaussian noise function
def add_gaussian_noise(image, mean=0, sigma=25):
    """add gaussian noise to image"""
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# process each image
for image_file in image_files:
    img_path = os.path.join(input_folder, image_file)
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Failed to read {image_file}, skipping")
        continue
    
    noisy_img = add_gaussian_noise(img)
    
    # save the noisy image
    output_path = os.path.join(output_folder, "noisy_" + image_file)
    cv2.imwrite(output_path, noisy_img)

    print(f"Done: {image_file} -> {output_path}")

print("All images processed!")
