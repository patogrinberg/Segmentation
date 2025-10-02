# -*- coding: utf-8 -*-
"""
PCA processing of images with prior alignment.
Original cropped images and PCA-transformed images are saved, preserving original filenames.
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os
from PIL import Image
import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# === Set your image folder path here ===
# Replace the string below with the path to your image directory
IMAGE_FOLDER = pathlib.Path("YOUR/IMAGE/FOLDER/PATH/HERE")
Image.MAX_IMAGE_PIXELS = 933120000
image_paths = sorted([file for file in IMAGE_FOLDER.glob('*.jpg')])

def align_images(image_files):
    """
    Aligns a list of images using the first image as reference.
    Uses ECC (Enhanced Correlation Coefficient) for translation-based alignment.
    """
    reference_img = plt.imread(image_files[0])
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_RGB2GRAY)
    height, width = reference_gray.shape
    aligned_images = [reference_img]

    for file_path in image_files[1:]:
        current_img = plt.imread(file_path)
        current_gray = cv2.cvtColor(current_img, cv2.COLOR_RGB2GRAY)
        current_gray = cv2.resize(current_gray, (width, height))
        current_img = cv2.resize(current_img, (width, height))

        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-6)

        try:
            _, warp_matrix = cv2.findTransformECC(reference_gray, current_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
            aligned_img = cv2.warpAffine(current_img, warp_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
            aligned_images.append(aligned_img)
        except cv2.error as error:
            print(f"Error aligning {file_path.name}: {error}")
            aligned_images.append(current_img)

    return aligned_images

# Align all images
aligned_images = align_images(image_paths)

# Determine minimum common size across all aligned images
min_height = min(img.shape[0] for img in aligned_images)
min_width = min(img.shape[1] for img in aligned_images)

# Divide images into grid quadrants
num_quadrants = 5
height_splits = np.linspace(0, min_height, num=num_quadrants + 1).astype(int)
width_splits = np.linspace(0, min_width, num=num_quadrants + 1).astype(int)

# Train PCA on the central quadrant
center_row = center_col = 2
channel_matrices = []
for img in aligned_images:
    for channel in range(3):
        crop = img[height_splits[center_row]:height_splits[center_row+1],
                   width_splits[center_col]:width_splits[center_col+1], channel]
        channel_matrices.append(crop)

channel_matrices = np.array(channel_matrices)
crop_shape = channel_matrices[0].shape
flattened_data = [channel.flatten() for channel in channel_matrices]
data_matrix = np.stack(flattened_data, axis=1)

# Normalize and apply PCA
scaler = StandardScaler().fit(data_matrix)
normalized_data = scaler.transform(data_matrix)
pca_model = PCA(n_components=3).fit(normalized_data)

# Create output directories
output_pca_dir = IMAGE_FOLDER / "PCA_m1"
output_cropped_dir = IMAGE_FOLDER / "Original_Cropped"
output_pca_dir.mkdir(parents=True, exist_ok=True)
output_cropped_dir.mkdir(parents=True, exist_ok=True)

# Apply PCA and save cropped and PCA images
for row in range(num_quadrants):
    for col in range(num_quadrants):
        quadrant_channels = []
        for img, path in zip(aligned_images, image_paths):
            crop = img[height_splits[row]:height_splits[row+1],
                       width_splits[col]:width_splits[col+1], :]
            base_name = path.stem
            crop_filename = f"{base_name}_q{row+1}{col+1}.jpg"
            plt.imsave(output_cropped_dir / crop_filename, crop, dpi=12000)

            for channel in range(3):
                quadrant_channels.append(crop[:, :, channel])

        quadrant_channels = np.array(quadrant_channels)
        crop_shape = quadrant_channels[0].shape
        flattened_data = [channel.flatten() for channel in quadrant_channels]
        data_matrix = np.stack(flattened_data, axis=1)
        normalized_data = scaler.transform(data_matrix)
        reduced_data = pca_model.transform(normalized_data)

        pca_channels = np.split(reduced_data, 3, axis=1)
        pca_image = np.moveaxis([channel.reshape(crop_shape) for channel in pca_channels], 0, 2)
        pca_image = (pca_image - np.min(pca_image)) / (np.max(pca_image) - np.min(pca_image))

        output_name = f"img_PCA_{row+1}{col+1}"
        plt.imsave(output_pca_dir / f"{output_name}.jpg", pca_image, dpi=12000)
        np.save(output_pca_dir / f"{output_name}.npy", pca_image)

print("✅ Processing complete. Cropped and PCA-transformed images saved with preserved filenames.")