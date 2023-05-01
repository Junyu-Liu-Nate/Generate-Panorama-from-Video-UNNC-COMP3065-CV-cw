import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

test_name = 'test_video_1'

vid_path = 'vid/' + test_name + '.mp4'
img_path = 'img/' + test_name + '.jpg'

# Open the video file
cap = cv2.VideoCapture(vid_path)
# Get the total number of frames in the video
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f'Total number of frames in the video: {frame_count}')

# Read the first frame from the video
ret, prev_frame = cap.read()

# # Create an empty panorama image
# h, w = prev_frame.shape[:2]
# panorama = np.zeros((h, w * 2, 3), np.uint8)
# panorama[:, :w] = prev_frame

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
        print(f'Processing frame {current_frame} of {frame_count}...')

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
        # Match the keypoints from the previous and current frames
        matches = bf.match(prev_des, des)
        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        src_pts = np.float32([ prev_kp[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        #----- Specify offset setting and calculate homography -----#
        w = frame.shape[1]
        h = frame.shape[0]

        # offset_x = w//3*2
        # offset_y = h//3
        offset_x = w//1*2
        offset_y = h//1

        dst_pts1 = dst_pts.copy()
        dst_pts1[:,:,0] = dst_pts1[:,:,0]+offset_x
        dst_pts1[:,:,1] = dst_pts1[:,:,1]+offset_y
        MM, mask = cv2.findHomography(src_pts, dst_pts1, cv2.RANSAC,5.0)
        panorama = cv2.warpPerspective(prev_frame, MM, (w+offset_x, h+offset_y))

        #----- Sharpen the image to remove blur incurred by homography -----#
        # # Create a sharpening kernel
        # kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        # # Apply the sharpening kernel to the image
        # panorama = cv2.filter2D(panorama, -1, kernel)

        #----- Add the new image at the lower right of the panaroma and blend it -----#
        for i in range(h):
            for j in range(w):
                # print(panorama[i+offset_y][j+offset_x])
                if panorama[i+offset_y][j+offset_x].all()==0:
                    panorama[i+offset_y][j+offset_x] = frame[i][j]
                elif frame[i][j].any()!=0:
                    panorama[i+offset_y][j+offset_x] = panorama[i+offset_y][j+offset_x]/2+frame[i][j]/2

        frame_folder_path = 'img/' + test_name + '/'
        if not os.path.exists(frame_folder_path):
            os.makedirs(frame_folder_path)
        cv2.imwrite(frame_folder_path + str(current_frame) + '.jpg', panorama)

        #----- Update the previous frame and keypoints -----#
        prev_frame = panorama.copy()
        sift = cv2.xfeatures2d.SIFT_create()
        prev_kp, prev_des = sift.detectAndCompute(prev_frame, None)
        # prev_kp = kp
        # prev_des = des
    else:
        print('Finished processing all frames')
        break

cv2.imwrite(img_path, panorama)

# Release the video file and destroy all windows
cap.release()
cv2.destroyAllWindows()
