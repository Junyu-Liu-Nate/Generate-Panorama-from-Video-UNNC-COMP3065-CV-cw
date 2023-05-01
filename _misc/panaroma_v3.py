import cv2
import numpy as np
import matplotlib.pyplot as plt

test_name = 'video_1'

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

current_frame = 0
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    # If the frame was read correctly
    if ret:
        # Increment the frame count
        current_frame += 1
        print(f'Processing frame {current_frame} of {frame_count}...')

        kp, des = sift.detectAndCompute(frame, None)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(prev_des, des, k=2)

        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)

        src_pts = np.float32([ prev_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

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

        for i in range(h):
            for j in range(w):
                # print(panorama[i+offset_y][j+offset_x])
                if panorama[i+offset_y][j+offset_x].all()==0:
                    panorama[i+offset_y][j+offset_x] = frame[i][j]
                elif frame[i][j].any()!=0:
                    panorama[i+offset_y][j+offset_x] = panorama[i+offset_y][j+offset_x]/2+frame[i][j]/2
        cv2.imwrite('img/frames/' + str(current_frame) + '.jpg', panorama)

        # Update the previous frame and keypoints
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
