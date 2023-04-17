import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from helpers import detect_content_range

def panorama(test_name, offset_x, offset_y, if_write_frame):
    vid_path = 'vid/' + test_name + '.mp4'
    left_img_path = 'img/' + test_name + '_left.jpg'
    right_img_path = 'img/' + test_name + '_right.jpg'

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
    panorama_right = np.zeros((h+offset_y, w+offset_x, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            panorama_right[i][j] = prev_frame[i][j]

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
        panorama_right = cv2.warpPerspective(frame, MM, (w+offset_x, h+offset_y))

        w, h = detect_content_range(prev_frame)

        ### Add the previoys frame at the upper left corner and blend it with the current frame
        mask1 = (panorama_right[:h,:w] == 0).all(axis=2) & (prev_frame[:h,:w] != 0).any(axis=2)
        panorama_right[:h,:w][mask1] = prev_frame[:h,:w][mask1]
        mask2 = (panorama_right[:h,:w] != 0).any(axis=2) & (prev_frame[:h,:w] != 0).any(axis=2)
        panorama_right[:h,:w][mask2] = prev_frame[:h,:w][mask2] / 2 + panorama_right[:h,:w][mask2] / 2

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
            panorama_left[offset_y+i][offset_x+j] = prev_frame[i][j]
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
        mask1 = (panorama_left[h_:h+offset_y, w_:w+offset_x] == 0).all(axis=2) & (prev_frame[h_:h+offset_y, w_:w+offset_x] != 0).any(axis=2)
        panorama_left[h_:h+offset_y, w_:w+offset_x][mask1] = prev_frame[h_:h+offset_y, w_:w+offset_x][mask1]
        mask2 = (panorama_left[h_:h+offset_y, w_:w+offset_x] != 0).any(axis=2) & (prev_frame[h_:h+offset_y, w_:w+offset_x] != 0).any(axis=2)
        panorama_left[h_:h+offset_y, w_:w+offset_x][mask2] = prev_frame[h_:h+offset_y, w_:w+offset_x][mask2] / 2 + panorama_left[h_:h+offset_y, w_:w+offset_x][mask2] / 2

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


### Choices for test_name: ['test_video_1', 'test_video_2', 'test_video_3'] ### 
test_name = 'test_video_1'

h, w = [1080, 1920]
offset_x = int(w*2)
offset_y = int(h*0.5)

### Flag to determine whether whether write frame-by-frame results when stitching
if_write_frame = True

panorama(test_name, offset_x, offset_y, if_write_frame)