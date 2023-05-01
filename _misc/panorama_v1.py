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

# Convert the frame to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

# Create an empty panorama image
h, w = prev_frame.shape[:2]
panorama = np.zeros((h, w * 2, 3), np.uint8)
panorama[:, :w] = prev_frame

# Create a SIFT object to detect and compute keypoints and descriptors
sift = cv2.xfeatures2d.SIFT_create()

# Find the keypoints and descriptors in the first frame
prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

# Create a BFMatcher object to match the keypoints
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)

current_frame = 0
while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    # print(ret, frame)

    # If the frame was read correctly
    if ret:
        # Increment the frame count
        current_frame += 1
        print(f'Processing frame {current_frame} of {frame_count}...')

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find the keypoints and descriptors in the frame
        kp, des = sift.detectAndCompute(gray, None)

        # Match the keypoints from the previous and current frames
        matches = bf.match(prev_des, des)

        # Sort the matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract the matched keypoints
        src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Warp the panorama image
        panorama = cv2.warpPerspective(panorama, M, (panorama.shape[1], panorama.shape[0]))
        cv2.imwrite('img/frames/' + str(current_frame) + '.jpg', panorama)

        # Update the previous frame and keypoints
        prev_frame = frame.copy()
        prev_gray = gray.copy()
        prev_kp = kp
        prev_des = des
    else:
        print('Finished processing all frames')
        break

panorama_rgb=cv2.cvtColor(panorama,cv2.COLOR_BGR2RGB)

cv2.imwrite(img_path, panorama)

# Release the video file and destroy all windows
cap.release()
cv2.destroyAllWindows()