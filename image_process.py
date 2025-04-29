import os
import cv2
import shutil
from pathlib import Path
import numpy as np
import torchvision.transforms as transforms
import torch
import random

image_size = 640

def clahe(img: np.array) -> np.array:
    """
    Enhance the input image using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    and resize the image to 512x512 resolution if necessary.
    
    Args:
        img (numpy.ndarray): Input image array in BGR format
        
    Returns:
        numpy.ndarray: Enhanced image array in BGR format
    """
    # Check if resizing is needed
    h, w = img.shape[:2]
    if h < image_size or w < image_size:
        # Calculate scaling factor to make the shorter side 512
        scale = image_size / min(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Resize to image_size  (cropping or padding if necessary)
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # Convert to LAB color space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)

    # Merge channels and convert back to BGR
    lab = cv2.merge((l2, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced


def reduce_quality(img: np.array, contrast_factor=0.7, noise_std=25) -> np.array:
    """
    Reduce image quality by decreasing contrast and adding grayscale Gaussian noise
    
    Args:
        img (numpy.ndarray): Input image array in BGR format
        contrast_factor (float): Factor to reduce contrast, range (0, 1)
        noise_std (float): Standard deviation of Gaussian noise, higher value means more noise
        
    Returns:
        numpy.ndarray: Degraded image array in BGR format
    """
    # Reduce contrast
    mean = np.mean(img, axis=(0, 1))
    reduced = img * contrast_factor + mean * (1 - contrast_factor)
    reduced = reduced.astype(np.uint8)

    # Generate 2D grayscale Gaussian noise
    noise_2d = np.random.normal(0, noise_std, img.shape[:2]).astype(np.int16)

    # Expand 2D noise to three channels
    noise = np.stack([noise_2d] * 3, axis=-1)

    # Add noise to image
    noisy = cv2.add(reduced.astype(np.int16), noise)

    # Ensure pixel values are within valid range [0, 255]
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return noisy


def process_dataset(src_root: str, dst_root: str, process) -> None:
    """
    Process the entire dataset by enhancing images and saving them to a new location.
    
    Iterates through all .jpg images in the traindata and valdata folders of the source directory,
    enhances each image, and saves both the enhanced images and their corresponding
    annotation files (.txt) to the appropriate folders in the target directory.
    
    Args:
        src_root (str): Path to the source dataset root directory
        dst_root (str): Path to the target dataset root directory
        process (function): Process function
        
    Note:
        - Target directory structure will be created automatically
        - Images that cannot be read will be skipped with a warning message
        - Annotation files will be copied directly to the new location
    """
    # Ensure target directory exists
    os.makedirs(dst_root, exist_ok=True)

    # Process traindata and valdata
    for dataset_type in ['traindata', 'valdata']:
        src_path = os.path.join(src_root, dataset_type)
        dst_path = os.path.join(dst_root, dataset_type)
        os.makedirs(dst_path, exist_ok=True)

        # Iterate through all image files
        for img_file in Path(src_path).glob('*.jpg'):
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"Can't read image: {img_file}")
                continue

            # Process image - can use either clahe or reduce_quality
            processed_img = process(img)

            # Save enhanced image
            dst_img_path = os.path.join(dst_path, img_file.name)
            cv2.imwrite(dst_img_path, processed_img)

            # Copy corresponding annotation file
            txt_file = img_file.with_suffix('.txt')
            if txt_file.exists():
                dst_txt_path = os.path.join(dst_path, txt_file.name)
                shutil.copy2(str(txt_file), dst_txt_path)
            else:
                print(f"Annotation file not found: {txt_file}")


if __name__ == '__main__':
    src_root = 'dataset-brain-tumor/uncategorized-reduced'
    dst_root = 'dataset-brain-tumor/uncategorized-reduced-clahe'

    process_dataset(src_root, dst_root, clahe)
    print("Dataset processing done.")
