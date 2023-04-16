import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# def cylindricalWarp(img):
#     """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
#     h, w = img.shape[:2]
#     K = np.array([[3773.33, 0, w / 2], [0, 3204.55, h / 2], [0, 0, 1]])

#     h_, w_ = img.shape[:2]
#     # pixel coordinates
#     y_i, x_i = np.indices((h_, w_))
#     X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
#     Kinv = np.linalg.inv(K)
#     X = Kinv.dot(X.T).T  # normalized coords
#     # calculate cylindrical coords (sin\theta, h, cos\theta)
#     A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
#     B = K.dot(A.T).T  # project back to image-pixels plane
#     # back from homog coords
#     B = B[:, :-1] / B[:, [-1]]
#     # make sure warp coords only within image bounds
#     B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
#     B = B.reshape(h_, w_, -1)

#     # img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders...
#     # warp the image according to cylindrical coords
#     return cv2.remap(img, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

def cylindricalWarp(img, K):
    # Get the dimensions of the image
    h, w = img.shape[:2]

    # Compute the focal length
    f = K[0, 0]

    # Create an array to store the cylindrical projection of the image
    cyl = np.zeros_like(img)

    # Iterate over the pixels of the cylindrical projection
    for y in range(h):
        for x in range(w):
            # Compute the cylindrical coordinates
            theta = (x - w / 2) / f
            h_ = (y - h / 2)

            # Compute the Cartesian coordinates
            x_ = f * np.tan(theta) + w / 2
            y_ = f * h_ / np.sqrt(f**2 + (x - w / 2)**2) + h / 2

            # Check if the coordinates are within the bounds of the image
            if 0 <= x_ < w and 0 <= y_ < h:
                # Map the pixel from the input image to the cylindrical projection
                cyl[y, x] = img[int(y_), int(x_)]

    return cyl


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

h, w = prev_frame.shape[:2]
K = np.array([[3773.33, 0, w / 2], [0, 3204.55, h / 2], [0, 0, 1]])
prev_frame = cylindricalWarp(prev_frame, K)

offset_x = w//1*2
offset_y = h//2
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
        print(f'Processing frame {current_frame} of {frame_count}...')

        h, w = frame.shape[:2]
        K = np.array([[3773.33, 0, w / 2], [0, 3204.55, h / 2], [0, 0, 1]])
        frame = cylindricalWarp(frame, K)

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

        offset_x = w//1*2
        offset_y = h//2

        # dst_pts1 = dst_pts.copy()
        # dst_pts1[:,:,0] = dst_pts1[:,:,0]+offset_x
        # dst_pts1[:,:,1] = dst_pts1[:,:,1]+offset_y
        MM, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        # panorama = cv2.warpPerspective(prev_frame, MM, (w+offset_x, h+offset_y))
        ### Should warp the current frame
        panorama = cv2.warpPerspective(frame, MM, (w+offset_x, h+offset_y))

        #----- Add the new image at the lower right of the panaroma and blend it -----#
        # for i in range(h):
        #     for j in range(w):
        #         # print(panorama[i+offset_y][j+offset_x])
        #         if panorama[i+offset_y][j+offset_x].all()==0:
        #             panorama[i+offset_y][j+offset_x] = frame[i][j]
        #         elif frame[i][j].any()!=0:
        #             panorama[i+offset_y][j+offset_x] = panorama[i+offset_y][j+offset_x]/2+frame[i][j]/2

        # panorama = cylindricalWarp(panorama)

        w = prev_frame.shape[1]
        h = prev_frame.shape[0]
        for i in range(h):
            for j in range(w):
                if panorama[i][j].all() == 0 and prev_frame[i][j].any() != 0:
                    panorama[i][j] = prev_frame[i][j]
                elif panorama[i][j].any() != 0 and prev_frame[i][j].any() != 0:
                    panorama[i][j] = prev_frame[i][j] / 2 + panorama[i][j] / 2

        # panorama = cv2.add(prev_frame, frame)

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
