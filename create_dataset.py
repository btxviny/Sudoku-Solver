import cv2
import os
import numpy as np
import pickle
import glob
import time
from detect_border import detect_border

# Define folders
image_save_folder = "./data/valid"
points_save_folder = "./data/vertices"
non_valid_folder = "./data/non_valid"

# Ensure folders exist
os.makedirs(image_save_folder, exist_ok=True)
os.makedirs(points_save_folder, exist_ok=True)
os.makedirs(non_valid_folder, exist_ok=True)

# Load all image files (change the extension if needed)
files = glob.glob("./data/not_annotated/*.jpeg")  # Modify path accordingly

# Track saved files for undo functionality
saved_files = []

cv2.namedWindow('image_with_contours', cv2.WINDOW_NORMAL)

for file in files:
    image = cv2.imread(file)
    image_with_contours, approx = detect_border(image)
    base_name = os.path.splitext(os.path.basename(file))[0]

    # If no valid Sudoku grid is found, save it in "non_valid"
    if approx is None:
        non_valid_path = os.path.join(non_valid_folder, f"{base_name}.jpg")
        cv2.imwrite(non_valid_path, image)
        print(f"Saved non-valid image for manual annotation: {non_valid_path}")
        continue  # Skip to the next image
    
    # Calculate contour area percentage
    contour_area = cv2.contourArea(approx) / (image.shape[0] * image.shape[1]) * 100
    cv2.putText(image_with_contours, f"{contour_area:.2f}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # If contour area is less than 40%, move to "non_valid"
    if contour_area < 40:
        non_valid_path = os.path.join(non_valid_folder, f"{base_name}.jpg")
        cv2.imwrite(non_valid_path, image)
        print(f"Contour too small ({contour_area:.2f}%). Moved to non-valid: {non_valid_path}")
        continue  # Skip to the next image
    cv2.imshow('image_with_contours',image_with_contours)
    time.sleep(1/30)
    # Save valid images automatically
    unique_name = f"{base_name}_{time.time()}"
    image_path = os.path.join(image_save_folder, unique_name + ".jpg")
    points_path = os.path.join(points_save_folder, unique_name + ".dat")

    cv2.imwrite(image_path, image)
    with open(points_path, "wb") as f:
        pickle.dump(approx, f)

    print(f"Saved valid image: {image_path} and points: {points_path}")
    saved_files.append((image_path, points_path))

cv2.destroyAllWindows()
