import cv2

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