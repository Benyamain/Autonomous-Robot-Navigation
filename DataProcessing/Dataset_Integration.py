import os
import shutil

# define folder groups: key = target folder name, value = list of source folders
folder_groups = {
    'depth_new': ['depth', 'depth_1', 'depth_2'],
    'images_new': ['images', 'images_1', 'images_2'],
    'labels_new': ['labels', 'labels_1', 'labels_2'],
    'scan_new': ['scan', 'scan_1', 'scan_2']
}

# base directory (can be changed to an absolute path)
base_dir = os.getcwd()

# merge each folder group
for target_folder, source_folders in folder_groups.items():
    target_path = os.path.join(base_dir, target_folder)
    os.makedirs(target_path, exist_ok=True)

    name_count = {}  # track filename conflicts

    for folder in source_folders:
        full_folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(full_folder_path):
            print(f"Skipping non-existent folder: {folder}")
            continue

        for filename in os.listdir(full_folder_path):
            src_file = os.path.join(full_folder_path, filename)
            if not os.path.isfile(src_file):
                continue  # skip subfolders

            name, ext = os.path.splitext(filename)

            # handle duplicate filenames
            if name not in name_count:
                name_count[name] = 0
                new_filename = f"{name}{ext}"
            else:
                name_count[name] += 1
                new_filename = f"{name}_{name_count[name]}{ext}"

            dst_file = os.path.join(target_path, new_filename)
            shutil.copy2(src_file, dst_file)

    print(f"Finished merging into {target_folder}")

print("All folder merging completed.")
