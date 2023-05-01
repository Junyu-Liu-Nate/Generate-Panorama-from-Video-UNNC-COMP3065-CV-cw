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

    # Straighten the panorama using perspective projection
    # (h, w) = cropped_image.shape[:2]
    # src_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    # dst_points = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    # M = cv2.getPerspectiveTransform(src_points, dst_points)
    # warped_image = cv2.warpPerspective(cropped_image, M, (w, h))

    # Alternatively, straighten the panorama using cylindrical projection
    # (h, w) = cropped_image.shape[:2]
    # focal_length = w
    # map_x = np.zeros((h,w), np.float32)
    # map_y = np.zeros((h,w), np.float32)
    # for y in range(h):
    #     for x in range(w):
    #         theta = (x - w/2) / focal_length
    #         h_ = (y - h/2) / focal_length
    #         X = np.sin(theta)
    #         Y = h_
    #         Z = np.cos(theta)
    #         x_ = focal_length * X/Z + w/2
    #         y_ = focal_length * Y/Z + h/2
    #         map_x[y,x] = x_
    #         map_y[y,x] = y_
    # warped_image = cv2.remap(cropped_image,map_x,map_y,cv2.INTER_LINEAR)

    ### Another cylindrical projection example ###

    # K = np.array([[3773.33, 0, w / 2], [0, 3204.55, h / 2], [0, 0, 1]])
    # K = np.array([[4656.07, 0, w / 2], [0, 3510.00, h / 2], [0, 0, 1]])
    # h_, w_ = image.shape[:2]
    # # pixel coordinates
    # y_i, x_i = np.indices((h_, w_))
    # X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
    # Kinv = np.linalg.inv(K)
    # X = Kinv.dot(X.T).T  # normalized coords
    # # calculate cylindrical coords (sin\theta, h, cos\theta)
    # A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    # B = K.dot(A.T).T  # project back to image-pixels plane
    # # back from homog coords
    # B = B[:, :-1] / B[:, [-1]]
    # # make sure warp coords only within image bounds
    # B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
    # B = B.reshape(h_, w_, -1)

    # img_rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)  # for transparent borders...
    # warp the image according to cylindrical coords
    # warped_image = cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

    return cropped_image

def cylindricalWarp(img):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h, w = img.shape[:2]
    ### focal length in pixels (x) = (26mm) * (1920 pixels) / (10.67mm) = 4656.07 pixels ###
    ### focal length in pixels (y) = (26mm) * (1080 pixels) / (8.0mm) = 3510.00 pixels ###
    K = np.array([[4656.07, 0, w / 2], [0, 3510.00, h / 2], [0, 0, 1]])

    h_, w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
    Kinv = np.linalg.inv(K)
    X = Kinv.dot(X.T).T  # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    B = K.dot(A.T).T  # project back to image-pixels plane
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]
    # make sure warp coords only within image bounds
    B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
    B = B.reshape(h_, w_, -1)

    img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img_rgba, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

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


def postprocess(test_name):
    # test_name = 'test_video_1'
    img_path = '../img/' + test_name + '_final.jpg'
    
    crop_img_path = '../img/' + test_name + '_crop.jpg'
    tight_crop_img_path = '../img/' + test_name + '_tightcrop.jpg'

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
    print('Finish tight cropping the final image (no black outliers).')