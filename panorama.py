import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from helpers import detect_content_range

def panorama(test_name, offset_x, offset_y):
    vid_path = 'vid/' + test_name + '.mp4'
    img_path = 'img/' + test_name + '.jpg'

    # Open the video file
    cap = cv2.VideoCapture(vid_path)
    # Get the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total number of frames in the video: {frame_count}')

    # Read the first frame from the video
    ret, prev_frame = cap.read()

    ### Create panorama
    h, w = prev_frame.shape[:2]
    # offset_x = w*3
    # offset_y = h*1.5
    panorama = np.zeros((h+offset_y, w+offset_x, 3), np.uint8)
    for i in range(h):
        for j in range(w):
            panorama[i][j] = prev_frame[i][j]

    # Create a SIFT object to detect and compute keypoints and descriptors
    sift = cv2.xfeatures2d.SIFT_create()
    # Find the keypoints and descriptors in the first frame
    prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)

    # Create a BFMatcher object to match the keypoints
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

    current_frame = 0
    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was read correctly
        if ret:
            # Increment the frame count
            current_frame += 1
            # Start the timer
            start_time = time.time()

            h, w = frame.shape[:2]
            K = np.array([[3773.33, 0, w / 2], [0, 3204.55, h / 2], [0, 0, 1]])

            #----- Calculate the matching points -----#
            kp, des = sift.detectAndCompute(frame, None)

            ### Match method 2
            # Match the keypoints from the previous and current frames
            matches = bf.match(prev_des, des)
            # Sort the matches by distance
            matches = sorted(matches, key=lambda x: x.distance)

            dst_pts = np.float32([ prev_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
            src_pts = np.float32([ kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

            #----- Specify offset setting and calculate homography -----#
            w = frame.shape[1]
            h = frame.shape[0]

            MM, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            panorama = cv2.warpPerspective(frame, MM, (w+offset_x, h+offset_y))

            w, h = detect_content_range(prev_frame)

            ### Add the previoys frame at the upper left corner and blend it with the current frame
            mask1 = (panorama[:h,:w] == 0).all(axis=2) & (prev_frame[:h,:w] != 0).any(axis=2)
            panorama[:h,:w][mask1] = prev_frame[:h,:w][mask1]
            mask2 = (panorama[:h,:w] != 0).any(axis=2) & (prev_frame[:h,:w] != 0).any(axis=2)
            panorama[:h,:w][mask2] = prev_frame[:h,:w][mask2] / 2 + panorama[:h,:w][mask2] / 2

            frame_folder_path = 'img/' + test_name + '/'
            if not os.path.exists(frame_folder_path):
                os.makedirs(frame_folder_path)
            cv2.imwrite(frame_folder_path + str(current_frame) + '.jpg', panorama)

            #----- Update the previous frame and keypoints -----#
            prev_frame = panorama.copy()
            sift = cv2.xfeatures2d.SIFT_create()
            prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
            
            # Stop the timer
            end_time = time.time()
            # Calculate and print the elapsed time
            elapsed_time = end_time - start_time
            print(f'Finish processing frame {current_frame} of {frame_count}..........Process time: {elapsed_time:.2f} seconds')
        else:
            print('Finished processing all frames')
            break

    cv2.imwrite(img_path, panorama)

    # Release the video file and destroy all windows
    cap.release()
    cv2.destroyAllWindows()
