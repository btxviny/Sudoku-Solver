import cv2
import os
import numpy as np
import pickle
import glob
import time
from detect_border import detect_border

# Define folders
image_save_folder = "./data/original/valid_images"
points_save_folder = "./data/original/vertices"
masks_save_folder = "./data/original/masks"

# Ensure folders exist
os.makedirs(image_save_folder, exist_ok=True)
os.makedirs(points_save_folder, exist_ok=True)
os.makedirs(masks_save_folder, exist_ok=True)


# Load all image files (change the extension if needed)
files = glob.glob("./data/original/images/*.jpeg") + glob.glob("./data/dataset/images/*.jpg")  # Modify path accordingly

# Track saved files for undo functionality
saved_files = []

cv2.namedWindow('image_with_contours', cv2.WINDOW_NORMAL)

for file in files:
    image = cv2.imread(file)
    image_with_contours, approx = detect_border(image)
    base_name = os.path.splitext(os.path.basename(file))[0]

    # If no valid Sudoku grid is found, save it in "non_valid"
    if approx is None:
        continue  # Skip to the next image
    
    # Calculate contour area percentage
    contour_area = cv2.contourArea(approx) / (image.shape[0] * image.shape[1]) * 100
    cv2.putText(image_with_contours, f"{contour_area:.2f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # If contour area is less than 40%, move to "non_valid"
    if contour_area < 40:
        continue  # Skip to the next image
    cv2.imshow('image_with_contours',image_with_contours)
    # Save valid images automatically
    h,w,_ = image.shape
    mask = np.zeros((h,w),dtype=np.float32)
    cv2.fillPoly(mask, [approx], 1)
    unique_name = f"{base_name}_{time.time()}"
    image_path = os.path.join(image_save_folder, unique_name + ".jpg")
    mask_path = os.path.join(masks_save_folder, unique_name + ".png")
    points_path = os.path.join(points_save_folder, unique_name + ".npy")
    
    cv2.imwrite(image_path, image)
    cv2.imwrite(mask_path, mask)
    np.save(points_path, approx)

    print(f"Saved valid image: {image_path} and points: {points_path}")
    saved_files.append((image_path, points_path))

cv2.destroyAllWindows()
