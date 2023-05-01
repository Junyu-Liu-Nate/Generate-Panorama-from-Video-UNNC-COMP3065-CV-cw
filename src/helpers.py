import cv2
import numpy as np

def detect_content_range(image):
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Return the max width and height
    return w, h

def fill_outermost_pixels(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for the content area
    mask = gray > 0
    
    # Find the contours of the content area
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the outermost pixels
    outermost_mask = np.zeros_like(gray)
    for contour in contours:
        cv2.drawContours(outermost_mask, [contour], -1, 1, 1)
    
    # Set the outermost pixels to pure black
    img[outermost_mask.astype(bool)] = 0
    
    return img