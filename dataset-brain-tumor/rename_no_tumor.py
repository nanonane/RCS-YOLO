import os
import glob
import shutil

# Define source directory and starting number
source_dir = "categorized/no-tumor-original"
dest_dir = "categorized/no-tumor"
start_num = 3065

# Ensure directory exists
if not os.path.exists(source_dir):
    print(f"Directory {source_dir} does not exist")
    exit(1)

os.makedirs(dest_dir, exist_ok=True)

# Get all jpg files and sort them
jpg_files = sorted(glob.glob(os.path.join(source_dir, "*.jpg")))
print(f"Found {len(jpg_files)} jpg files")

# Copy files to destination directory and create empty txt files
for i, jpg_file in enumerate(jpg_files):
    # New filename
    new_num = start_num + i
    new_jpg = os.path.join(dest_dir, f"{new_num}.jpg")
    new_txt = os.path.join(dest_dir, f"{new_num}.txt")
    
    # Copy jpg file to destination directory
    shutil.copy2(jpg_file, new_jpg)
    print(f"Copied: {os.path.basename(jpg_file)} -> {new_num}.jpg")
    
    # Create empty txt file in destination directory
    with open(new_txt, 'w') as f:
        pass
    print(f"Created empty file: {new_num}.txt")

print(f"Processing completed, processed {len(jpg_files)} files in total")
