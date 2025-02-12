import cv2
import os 
import numpy as np


def adjust_brightness_contrast(image, alpha=1.2, beta=50):
    """
    Adjust the brightness and contrast of the image.
    alpha: contrast control, beta: brightness control
    """
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def clahe(image):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    return equalized

def gaussian_blur(image, kernel_size=(5, 5)):
    """
    Apply Gaussian blur to reduce noise and smooth the image.
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred

def adaptive_threshold(image):
    """
    Apply adaptive thresholding to handle varying lighting conditions.
    """
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, 11, 2)
    return adaptive_thresh

def preprocess_image(image):
    """
    Preprocess the image: adjust brightness/contrast, apply CLAHE, Gaussian blur, and adaptive thresholding.
    """
    adjusted_image = adjust_brightness_contrast(image, alpha=1.2, beta=50)  # Adjust brightness/contrast
    equalized_image = clahe(adjusted_image)  # Apply CLAHE for local contrast adjustment
    blurred_image = gaussian_blur(equalized_image)  # Gaussian blur to reduce noise
    thresholded_image = adaptive_threshold(blurred_image)  # Adaptive thresholding
    return thresholded_image

def detect_border(image):
    # Preprocess the image to enhance the grid visibility
    preprocessed_image = preprocess_image(image)
    
    # Apply Canny edge detection on the preprocessed image
    edges = cv2.Canny(preprocessed_image , 50, 150)  # Adjust Canny thresholds for better edge detection
    
    # Find contours (only external contours for the whole grid)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (descending order)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Draw contours on the original image
    image_with_contours = image.copy()
    
    # Filter contours based on area to avoid small cells or noise
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Ignore small contours, adjust this value based on your image size
            pts = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)  # Tighter epsilon
            if len(pts) == 4:  # Looking for a 4-point quadrilateral (the grid)
                # Draw a rectangle around the largest contour with 4 points
                cv2.polylines(image_with_contours, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                return image_with_contours, pts
    
    return image_with_contours, None