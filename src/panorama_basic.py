import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from helpers import detect_content_range, fill_outermost_pixels

def panorama_basic(test_name, offset_x, offset_y, if_write_frame):
    if not os.path.exists('../img/'):
        os.makedirs('../img/')

    vid_path = '../vid/' + test_name + '.mp4'
    left_img_path = '../img/' + test_name + '_left.jpg'
    right_img_path = '../img/' + test_name + '_right.jpg'
    final_img_path = '../img/' + test_name + '_final.jpg'

    ### Open the video file
    cap = cv2.VideoCapture(vid_path)
    ### Get the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total number of frames in the video: {frame_count}')

    #----- Separate frames to left_frames and right_frames -----#
    isValid = False
    while isValid == False:
        project_idx= input("Please enter the index for projection plane: ")
        project_idx = int(project_idx)
        if project_idx > 1 and project_idx < frame_count:
            isValid = True

    counter = 0
    left_frames = []
    right_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if counter <= project_idx:
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

        ### Match method
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
  
        mask1 = (panorama_right[:HEIGHT,:w] == 0).all(axis=2) & (prev_frame[:HEIGHT,:w] != 0).any(axis=2)
        mask2 = (panorama_right[:HEIGHT,:w] != 0).any(axis=2) & (prev_frame[:HEIGHT,:w] != 0).any(axis=2)
        panorama_right[:HEIGHT,:w][mask1] = prev_frame[:HEIGHT,:w][mask1]
        panorama_right[:HEIGHT,:w][mask2] = prev_frame[:HEIGHT,:w][mask2] / 2 + panorama_right[:HEIGHT,:w][mask2] / 2

        ### Write frame-by-frame result to folder
        if if_write_frame:
            frame_folder_path = '../img/' + test_name + '/'
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

    # cv2.imwrite(right_img_path, panorama_right)

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

        ### Match method
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
 
        mask1 = (panorama_left[:HEIGHT, w_:w+offset_x] == 0).all(axis=2) & (prev_frame[:HEIGHT, w_:w+offset_x] != 0).any(axis=2)
        panorama_left[:HEIGHT, w_:w+offset_x][mask1] = prev_frame[:HEIGHT, w_:w+offset_x][mask1]
        mask2 = (panorama_left[:HEIGHT, w_:w+offset_x] != 0).any(axis=2) & (prev_frame[:HEIGHT, w_:w+offset_x] != 0).any(axis=2)
        panorama_left[:HEIGHT, w_:w+offset_x][mask2] = prev_frame[:HEIGHT, w_:w+offset_x][mask2] / 2 + panorama_left[:HEIGHT, w_:w+offset_x][mask2] / 2

        ### Write frame-by-frame result to folder
        if if_write_frame:
            frame_folder_path = '../img/' + test_name + '/'
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

    # cv2.imwrite(left_img_path, panorama_left)

    #----- Stitch the left and right images together -----#
    # panorama_left = cv2.imread(left_img_path)
    # panorama_right = cv2.imread(right_img_path)

    panorama_left = fill_outermost_pixels(panorama_left)
    panorama_right = fill_outermost_pixels(panorama_right)

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

    panorama_final = fill_outermost_pixels(panorama_final)
    panorama_final = fill_outermost_pixels(panorama_final)
    panorama_final = fill_outermost_pixels(panorama_final)

    mask1 = (panorama_final[:HEIGHT,:WIDTH] == 0).all(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    panorama_final[:HEIGHT,:WIDTH][mask1] = panorama_left[:HEIGHT,:WIDTH][mask1]
    mask2 = (panorama_final[:HEIGHT,:WIDTH] <= 50).any(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    panorama_final[:HEIGHT,:WIDTH][mask2] = panorama_left[:HEIGHT,:WIDTH][mask2]
    # mask3 = (panorama_final[:HEIGHT,:WIDTH] != 0).any(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    # panorama_final[:HEIGHT,:WIDTH][mask3] = panorama_left[:HEIGHT,:WIDTH][mask3] / 2 + panorama_final[:HEIGHT,:WIDTH][mask3] / 2
    # panorama_final[:HEIGHT,:WIDTH] = panorama_left[:HEIGHT,:WIDTH]

    cv2.imwrite(final_img_path, panorama_final)