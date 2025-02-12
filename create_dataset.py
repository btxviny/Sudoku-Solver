import cv2
import os
import numpy as np
import pickle
import glob
import time
from detect_border import detect_border

# Define folders
image_save_folder = "data/valid"
points_save_folder = "data/vertices"
non_valid_folder = "data/non_valid"

# Ensure folders exist
os.makedirs(image_save_folder, exist_ok=True)
os.makedirs(points_save_folder, exist_ok=True)
os.makedirs(non_valid_folder, exist_ok=True)

# Load all image files (change the extension if needed)
files = glob.glob("./data/detection_data/*.jpeg")  # Modify path accordingly

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

    cv2.imshow('image_with_contours', image_with_contours)
    key = cv2.waitKey(0) & 0xFF  # Get key press

    if key == ord("f"):  # Save if 'f' is pressed
        # Generate a unique filename
        unique_name = f"{base_name}_{time.time()}"

        # Save valid image
        image_path = os.path.join(image_save_folder, unique_name + ".jpg")
        cv2.imwrite(image_path, image_with_contours)

        # Save points
        points_path = os.path.join(points_save_folder, unique_name + ".dat")
        with open(points_path, "wb") as f:
            pickle.dump(approx, f)

        print(f"Saved: {image_path} and {points_path}")
        saved_files.append((image_path, points_path))

    elif key == ord("s"):  # Skip image and move it to "non_valid"
        non_valid_path = os.path.join(non_valid_folder, f"{base_name}.jpg")
        cv2.imwrite(non_valid_path, image)
        print(f"Skipped and saved for manual annotation: {non_valid_path}")

    elif key == ord("u") and saved_files:  # Undo last save if 'u' is pressed
        last_image, last_points = saved_files.pop()
        os.remove(last_image)  # Remove last saved image
        os.remove(last_points)  # Remove last saved points file
        print(f"Undo: Removed {last_image} and {last_points}")

    elif key == ord("q"):  # Quit if 'q' is pressed
        print("Quitting... Deleting created files.")
        # Delete all files inside the created directories
        for folder in [image_save_folder, points_save_folder, non_valid_folder]:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)

        # Optionally, remove the empty directories
        os.rmdir(image_save_folder)
        os.rmdir(points_save_folder)
        os.rmdir(non_valid_folder)
        break
cv2.destroyAllWindows()
