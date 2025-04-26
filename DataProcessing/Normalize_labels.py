import os
from PIL import Image

# set folder paths
base_dir = os.getcwd()
label_input_dir = os.path.join(base_dir, 'labels_new')
image_dir = os.path.join(base_dir, 'images_new')  # used to get image sizes
label_output_dir = os.path.join(base_dir, 'label_clean')
os.makedirs(label_output_dir, exist_ok=True)

# process each txt file
for filename in os.listdir(label_input_dir):
    if not filename.endswith('.txt'):
        continue

    txt_path = os.path.join(label_input_dir, filename)
    image_name = filename.replace('.txt', '.jpg')
    image_path = os.path.join(image_dir, image_name)

    if not os.path.exists(image_path):
        print(f"Missing corresponding image: {image_name}, skipping")
        continue

    # get image size
    with Image.open(image_path) as img:
        w, h = img.size

    output_path = os.path.join(label_output_dir, filename)

    with open(txt_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # ignore malformed lines

            cls = parts[0]
            cx, cy, bw, bh = map(float, parts[1:5])

            # normalize
            cx /= w
            cy /= h
            bw /= w
            bh /= h

            outfile.write(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

print("All label files processed and saved to 'label_clean'.")
