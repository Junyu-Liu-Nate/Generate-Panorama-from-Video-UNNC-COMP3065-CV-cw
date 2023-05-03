import cv2
import numpy as np

def rectify_perspective(image):
     # Convert the image to grayscale and create a binary mask
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    # Find the largest contour in the mask and compute its bounding rectangle
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image using the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image


def get_contour_bounds(contour):
    # Find the coordinates of the contour points
    coords = np.squeeze(contour)

    # # # Find the minimum and maximum x and y coordinates
    # # x_min, y_min = coords.min(axis=0)
    # # x_max, y_max = coords.max(axis=0)

    # # Find the maximum x coordinate when y=0
    # x_max = coords[coords[:,1] == 0][:,0].max()
    # # Find the maximum y coordinate when x=0
    # y_max = coords[coords[:,0] == 0][:,1].max()

    # # Find the minimum x coordinate when y=0
    # x_min = coords[coords[:,1] == 0][:,0].min()
    # # Find the maximum y coordinate when x=0
    # y_min = coords[coords[:,0] == 0][:,1].min()

    x_values = [x for x, y in coords]
    y_values = [y for x, y in coords]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)

    return x_min, x_max, y_min, y_max

def get_tight_contour_bounds(contour, x_min, x_max, y_min, y_max):
    # Find the coordinates of the contour points
    coords = np.squeeze(contour)

    # x_values = [x for x, y in coords]
    # y_values = [y for x, y in coords]
    # x_min = min(x_values)
    # x_max = max(x_values)
    # y_min = min(y_values)
    # y_max = max(y_values)

    WIDTH = x_max - x_min
    HEIGHT = y_max - y_min

    y_values_upper = []
    y_values_lower = []
    x_values_left = []
    x_values_right = []
    for coord in coords:
        if coord[0] >= x_min + 0.25*WIDTH and coord[0] <= x_max - 0.25*WIDTH:
            if coord[1] >= y_min + 0.5*HEIGHT:
                y_values_lower.append(coord[1])
            elif coord[1] < y_min + 0.5*HEIGHT:
                y_values_upper.append(coord[1])
    
    y_min_tight = max(y_values_upper)
    y_max_tight = min(y_values_lower)

    return y_min_tight, y_max_tight

def crop_content(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x_min, x_max, y_min, y_max = get_contour_bounds(largest_contour)

    # Crop the image to the bounding box
    cropped = image[y_min:y_max,x_min:x_max]

    return cropped

def crop_content_tight(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find the contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x_min, x_max, y_min, y_max = get_contour_bounds(largest_contour)

    y_min_tight, y_max_tight = get_tight_contour_bounds(largest_contour, x_min, x_max, y_min, y_max)

    # Crop the image to the bounding box
    tight_cropped = image[y_min_tight:y_max_tight,x_min:x_max]

    return tight_cropped

def transform_perspective(img):
    # # Load the image
    # img = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Find the contours in the binary image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    max_contour = max(contours, key=cv2.contourArea)

    width = img.shape[1]
    height = img.shape[0]
    left_list = []
    right_list = []
    for point in max_contour:
        pnt = point[0]
        if pnt[0] <= width / 2:
            left_list.append(pnt)
        else:
            right_list.append(pnt)
    
    left_min = min(left_list, key=lambda point: point[0])
    left_max = max(left_list, key=lambda point: point[0])
    if left_min[1] >= left_max[1]:
        top_left = left_max
        bottom_left = left_min
    else:
        top_left = left_min
        bottom_left = left_max

    right_min = min(right_list, key=lambda point: point[0])
    right_max = max(right_list, key=lambda point: point[0])
    if right_min[1] >= right_max[1]:
        top_right = right_max
        bottom_right = right_min
    else:
        top_right = right_min
        bottom_right = right_max

    # Define the source points
    src_points = np.float32([top_left,top_right,bottom_left,bottom_right])

    # Define the destination points
    width = img.shape[1]
    height = img.shape[0]
    dst_points = np.float32([[0, 0], [width-1, 0], [0,height-1], [width-1,height-1]])

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_points,dst_points)

    # Warp the image
    warped_img = cv2.warpPerspective(img,M,(width,height))

    return warped_img


def postprocess(test_name):
    img_path = '../img/' + test_name + '_final.jpg'
    
    crop_img_path = '../img/' + test_name + '_crop.jpg'
    tight_crop_img_path = '../img/' + test_name + '_tightcrop.jpg'
    transform_img_path = '../img/' + test_name + '_transform.jpg'

    image = cv2.imread(img_path)
    # if image is None:
    #     print('Image not loaded')
    # else:
    #     print('Image loaded successfully')

    # rectified_img = rectify_perspective(image)
    # rectified_img = cylindricalWarp(image)
    
    crop_img = crop_content(image)
    cv2.imwrite(crop_img_path, crop_img)
    print('Finish cropping the final image (all content preserved).')

    tight_crop_img = crop_content_tight(image)
    cv2.imwrite(tight_crop_img_path, tight_crop_img)
    print('Finish tight cropping the final image (eliminate black outliers).')

    transform_img = transform_perspective(tight_crop_img)
    if 'test_video_4' in test_name:
        cv2.imwrite(transform_img_path, tight_crop_img)
    else:
        cv2.imwrite(transform_img_path, transform_img)
    print('Finish rectifying the perspective of the panorama.')

