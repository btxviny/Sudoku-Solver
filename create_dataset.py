import os
import random
import cv2
import numpy as np
from tqdm import tqdm
from loguru import logger
from concurrent.futures import ThreadPoolExecutor

# Function to apply random perspective transformation
def apply_perspective_transform(img):
    h, w = img.shape[:2]
    max_x_offset = int(w * 0.05)
    max_y_offset = int(h * 0.05)
    
    src_pts = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
    dst_pts = np.float32([
        [random.randint(0, max_x_offset), random.randint(0, max_y_offset)], 
        [w-1-random.randint(0, max_x_offset), random.randint(0, max_y_offset)], 
        [random.randint(0, max_x_offset), h-1-random.randint(0, max_y_offset)], 
        [w-1-random.randint(0, max_x_offset), h-1-random.randint(0, max_y_offset)]
    ])
    
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(img, M, (w, h))

# Function to apply random rotation
def apply_random_rotation(img):
    angle = random.randint(-10, 10)
    h, w = img.shape[:2]
    
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(img, M, (w, h))

# Function to add noise
def add_noise(image):
    h, w = image.shape
    noise = np.random.normal(0, 5, (h, w))
    return np.clip(image + noise, 0, 255).astype(np.uint8)

# Function to generate a digit image
def generate_digit_image(digit, size=(28, 28)):
    img = np.zeros((100, 100), dtype=np.uint8)  # Black background
    font = random.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_DUPLEX])
    
    thickness = random.randint(1, 3)
    font_scale = random.uniform(2, 3.5)
    
    text_size = cv2.getTextSize(str(digit), font, font_scale, thickness)[0]
    text_x = (img.shape[1] - text_size[0]) // 2
    text_y = (img.shape[0] + text_size[1]) // 2
    
    cv2.putText(img, str(digit), (text_x, text_y), font, font_scale, 255, thickness)
    
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = apply_random_rotation(img)
    img = apply_perspective_transform(img)
    img = add_noise(img)
    
    return img.astype(np.uint8)

def generate_empty_image(size=(28, 28)):
    img = np.zeros((100, 100), dtype=np.uint8)  # Black background
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = apply_random_rotation(img)
    img = apply_perspective_transform(img)
    img = add_noise(img)
    return img.astype(np.uint8)

# Function to save an image and return label data
def save_image_label(idx, images_path):
    #for every 10 digit images generate an empty image
    if random.randint(1,10) == 1:
        image = generate_empty_image()
        label = 0
    else:
        label = random.randint(1, 9)
        image = generate_digit_image(digit=label)
    image_path = f"{images_path}/{idx}.jpg"
    cv2.imwrite(image_path, image)
    
    return f"{image_path},{label}\n"  # Return the label instead of writing directly

# Main script with threading
if __name__ == "__main__":
    logger.info("Creating Digital Digits with Threading")
    
    images_path = 'digits/images'
    labels_txt_path = 'digits/labels.txt'
    os.makedirs(images_path, exist_ok=True)
    
    num_images = 60000
    num_threads = 10  # Adjust based on CPU cores

    # Generate images using threading and collect label data
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        labels = list(tqdm(executor.map(lambda idx: save_image_label(idx, images_path), range(num_images)), total=num_images))

    # Write all labels at once (thread-safe)
    with open(labels_txt_path, 'w') as f:
        f.writelines(labels)

    logger.info("Dataset generation completed!")
