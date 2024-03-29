import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from helpers import detect_content_range
from perspective import postprocess

def remove_lines(img, threshold=50):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute the gradient of the image
    grad_x, grad_y = np.gradient(gray)
    
    # Create masks for the vertical and horizontal lines
    mask_x = np.abs(grad_x) > threshold
    mask_y = np.abs(grad_y) > threshold
    
    # Find the indices of the vertical and horizontal lines
    idx_x = np.where(mask_x)
    idx_y = np.where(mask_y)
    
    # Replace the vertical lines with the average of their left and right neighbors
    for i, j in zip(*idx_x):
        left = img[i, max(0, j-1)]
        right = img[i, min(img.shape[1]-1, j+1)]
        img[i, j] = (left + right) / 2
    
    # Replace the horizontal lines with the average of their upper and lower neighbors
    for i, j in zip(*idx_y):
        up = img[max(0, i-1), j]
        down = img[min(img.shape[0]-1, i+1), j]
        img[i, j] = (up + down) / 2
    
    return img

def fill_outermost_pixels(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for the content area
    mask = gray > 0
    
    # Find the contours of the content area
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the outermost pixels of the content area
    outermost_pixels = set()
    for contour in contours:
        for i in range(-1, len(contour)-1):
            p1 = tuple(contour[i][0])
            p2 = tuple(contour[i+1][0])
            if cv2.clipLine((0, 0, img.shape[1]-1, img.shape[0]-1), p1, p2):
                line_mask = np.zeros_like(gray)
                cv2.line(line_mask, p1, p2, 1)
                outermost_pixels.update(set(zip(*np.where(line_mask))))
    
    # # Replace the outermost pixels with their nearest inner neighbor
    # for i, j in outermost_pixels:
    #     if mask[i, j]:
    #         neighbors = img[max(0, i-1):i+2, max(0, j-1):j+2]
    #         inner_neighbors = neighbors[~np.all(neighbors == 0, axis=-1)]
    #         if len(inner_neighbors) > 0:
    #             img[i, j] = np.mean(inner_neighbors, axis=0)

    # Set the outermost pixels to pure black
    for i, j in outermost_pixels:
        if mask[i, j]:
            img[i, j] = 0
    
    return img

def panorama(test_name, offset_x, offset_y, if_write_frame):
    vid_path = 'vid/' + test_name + '.mp4'
    left_img_path = 'img/' + test_name + '_left.jpg'
    right_img_path = 'img/' + test_name + '_right.jpg'
    final_img_path = 'img/' + test_name + '_final.jpg'

    ### Open the video file
    cap = cv2.VideoCapture(vid_path)
    ### Get the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total number of frames in the video: {frame_count}')

    #----- Separate frames to left_frames and right_frames -----#
    counter = 0
    left_frames = []
    right_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if counter <= frame_count / 2:
                left_frames.append(frame)
            else:
                right_frames.append(frame)
            counter += 1
        else:
            cap.release()
            break
    left_frames = left_frames[::-1]

    #----- Stitch the right video frames -----#
    prev_frame = right_frames[0]

    ### Create panorama
    h, w = prev_frame.shape[:2]
    WIDTH = w+offset_x
    HEIGHT = h+offset_y
    panorama_right = np.zeros((h+offset_y, w+offset_x, 3), np.uint8)
    for i in range(int(HEIGHT/2 - h/2), int(HEIGHT/2 + h/2)):
        for j in range(w):
            panorama_right[i][j] = prev_frame[i - int(HEIGHT/2 - h/2)][j]
    prev_frame = panorama_right

    ### Create a SIFT object to detect and compute keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    ### Find the keypoints and descriptors in the first frame
    prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)

    ### Create a BFMatcher object to match the keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    current_frame = 0
    for frame in right_frames:
        ### Increment the frame count
        current_frame += 1
        ### Start the timer
        start_time = time.time()

        #----- Calculate the matching points -----#
        kp, des = sift.detectAndCompute(frame, None)

        ### Match method 1
        # FLANN_INDEX_KDTREE = 0
        # index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        # search_params = dict(checks = 50)
        # flann = cv2.FlannBasedMatcher(index_params, search_params)
        # matches = flann.knnMatch(prev_des, des, k=2)
        # good = []
        # for m,n in matches:
        #     if m.distance < 0.7*n.distance:
        #         good.append(m)

        ### Match method 2
        ### Match the keypoints from the previous and current frames
        matches = bf.match(prev_des, des)
        ### Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        dst_pts = np.float32([ prev_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        src_pts = np.float32([ kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

        #----- Specify offset setting and calculate homography -----#
        w = frame.shape[1]
        h = frame.shape[0]

        MM, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        panorama_right = cv2.warpPerspective(frame, MM, (w+offset_x, h+offset_y))

        panorama_right = fill_outermost_pixels(panorama_right)

        w, h = detect_content_range(prev_frame)
        # w = w+offset_x
        # h = h+offset_y

        ### Add the previoys frame at the upper left corner and blend it with the current frame
        # for i in range(h):
        #     for j in range(w):
        #         if panorama[i][j].all() == 0 and prev_frame[i][j].any() != 0:
        #             panorama[i][j] = prev_frame[i][j]
        #         elif panorama[i][j].any() != 0 and prev_frame[i][j].any() != 0:
        #             panorama[i][j] = prev_frame[i][j] / 2 + panorama[i][j] / 2
        mask1 = (panorama_right[:HEIGHT,:w] == 0).all(axis=2) & (prev_frame[:HEIGHT,:w] != 0).any(axis=2)
        mask2 = (panorama_right[:HEIGHT,:w] != 0).any(axis=2) & (prev_frame[:HEIGHT,:w] != 0).any(axis=2)
        panorama_right[:HEIGHT,:w][mask1] = prev_frame[:HEIGHT,:w][mask1]
        panorama_right[:HEIGHT,:w][mask2] = prev_frame[:HEIGHT,:w][mask2] / 2 + panorama_right[:HEIGHT,:w][mask2] / 2

        # panorama_right = remove_lines(panorama_right)

        ### Write frame-by-frame result to folder
        if if_write_frame:
            frame_folder_path = 'img/' + test_name + '/'
            if not os.path.exists(frame_folder_path):
                os.makedirs(frame_folder_path)
            cv2.imwrite(frame_folder_path + str(current_frame) + '_right.jpg', panorama_right)

        #----- Update the previous frame and keypoints -----#
        prev_frame = panorama_right.copy()
        sift = cv2.xfeatures2d.SIFT_create()
        prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
        
        ### Stop the timer
        end_time = time.time()
        ### Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f'Finish processing right frame {current_frame} of {len(right_frames)}..........Process time: {elapsed_time:.2f} seconds')

    cv2.imwrite(right_img_path, panorama_right)

    #----- Stitch the left video frames -----#
    prev_frame = left_frames[0]

    ### Create panorama
    ### When further modifying it to align along the center line, adjust the first frame, and in the later loop only adjust the loop range
    h, w = prev_frame.shape[:2]
    panorama_left = np.zeros((h+offset_y, w+offset_x, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            panorama_left[int(HEIGHT/2 - h/2)+i][offset_x+j] = prev_frame[i][j]
    prev_frame = panorama_left

    ### Create a SIFT object to detect and compute keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    ### Find the keypoints and descriptors in the first frame
    prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)

    ### Create a BFMatcher object to match the keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    current_frame = 0
    for frame in left_frames:
        ### Increment the frame count
        current_frame += 1
        ### Start the timer
        start_time = time.time()

        #----- Calculate the matching points -----#
        kp, des = sift.detectAndCompute(frame, None)

        ### Match method 2
        ### Match the keypoints from the previous and current frames
        matches = bf.match(prev_des, des)
        ### Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        dst_pts = np.float32([ prev_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        src_pts = np.float32([ kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        #----- Specify offset setting and calculate homography -----#
        w = frame.shape[1]
        h = frame.shape[0]

        MM, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        panorama_left = cv2.warpPerspective(frame, MM, (w+offset_x, h+offset_y))

        panorama_left = fill_outermost_pixels(panorama_left)

        w_, h_ = detect_content_range(prev_frame)
        h_ = h + offset_y - h_
        w_ = w + offset_x - w_
        # print([h_, h+offset_y])
        # print([w_, w+offset_x])
        ### Add the previoys frame at the lower right corner and blend it with the current frame
        # for i in range(h_, h+offset_y):
        #     for j in range(w_, w+offset_x):
        #         if panorama_left[i][j].all() == 0 and prev_frame[i][j].any() != 0:
        #             panorama_left[i][j] = prev_frame[i][j]
        #         elif panorama_left[i][j].any() != 0 and prev_frame[i][j].any() != 0:
        #             panorama_left[i][j] = prev_frame[i][j] / 2 + panorama_left[i][j] / 2
        # mask1 = (panorama_left[h_:h+offset_y, w_:w+offset_x] == 0).all(axis=2) & (prev_frame[h_:h+offset_y, w_:w+offset_x] != 0).any(axis=2)
        # panorama_left[h_:h+offset_y, w_:w+offset_x][mask1] = prev_frame[h_:h+offset_y, w_:w+offset_x][mask1]
        # mask2 = (panorama_left[h_:h+offset_y, w_:w+offset_x] != 0).any(axis=2) & (prev_frame[h_:h+offset_y, w_:w+offset_x] != 0).any(axis=2)
        # panorama_left[h_:h+offset_y, w_:w+offset_x][mask2] = prev_frame[h_:h+offset_y, w_:w+offset_x][mask2] / 2 + panorama_left[h_:h+offset_y, w_:w+offset_x][mask2] / 2
        mask1 = (panorama_left[:HEIGHT, w_:w+offset_x] == 0).all(axis=2) & (prev_frame[:HEIGHT, w_:w+offset_x] != 0).any(axis=2)
        panorama_left[:HEIGHT, w_:w+offset_x][mask1] = prev_frame[:HEIGHT, w_:w+offset_x][mask1]
        mask2 = (panorama_left[:HEIGHT, w_:w+offset_x] != 0).any(axis=2) & (prev_frame[:HEIGHT, w_:w+offset_x] != 0).any(axis=2)
        panorama_left[:HEIGHT, w_:w+offset_x][mask2] = prev_frame[:HEIGHT, w_:w+offset_x][mask2] / 2 + panorama_left[:HEIGHT, w_:w+offset_x][mask2] / 2

        ### Write frame-by-frame result to folder
        if if_write_frame:
            frame_folder_path = 'img/' + test_name + '/'
            if not os.path.exists(frame_folder_path):
                os.makedirs(frame_folder_path)
            cv2.imwrite(frame_folder_path + str(current_frame) + '_left.jpg', panorama_left)

        #----- Update the previous frame and keypoints -----#
        prev_frame = panorama_left.copy()
        sift = cv2.xfeatures2d.SIFT_create()
        prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
        
        ### Stop the timer
        end_time = time.time()
        ### Calculate and print the elapsed time
        elapsed_time = end_time - start_time
        print(f'Finish processing left frame {current_frame} of {len(left_frames)}..........Process time: {elapsed_time:.2f} seconds')

    cv2.imwrite(left_img_path, panorama_left)

    #----- Stitch the left and right images together -----#
    panorama_left = cv2.imread(left_img_path)
    panorama_right = cv2.imread(right_img_path)

    sift = cv2.xfeatures2d.SIFT_create()
    prev_kp, prev_des = sift.detectAndCompute(panorama_left, None)

    kp, des = sift.detectAndCompute(panorama_right, None)

    ### Create a BFMatcher object to match the keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
    ### Match the keypoints from the previous and current frames
    matches = bf.match(prev_des, des)
    ### Sort the matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    dst_pts = np.float32([ prev_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    src_pts = np.float32([ kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

    MM, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    panorama_final = cv2.warpPerspective(panorama_right, MM, (WIDTH*2, HEIGHT))

    mask1 = (panorama_final[:HEIGHT,:WIDTH] == 0).all(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    panorama_final[:HEIGHT,:WIDTH][mask1] = panorama_left[:HEIGHT,:WIDTH][mask1]
    mask2 = (panorama_final[:HEIGHT,:WIDTH] != 0).any(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    panorama_final[:HEIGHT,:WIDTH][mask2] = panorama_left[:HEIGHT,:WIDTH][mask2] / 2 + panorama_final[:HEIGHT,:WIDTH][mask2] / 2

    cv2.imwrite(final_img_path, panorama_final)


### Choices for test_name: ['test_video_1', 'test_video_2', 'test_video_3'] ### 
test_name = 'test_video_1'

h, w = [1080, 1920]
offset_x = int(w*1.8)
offset_y = int(h*0.5)

### Flag to determine whether whether write frame-by-frame results when stitching
if_write_frame = True

panorama(test_name, offset_x, offset_y, if_write_frame)
postprocess(test_name)