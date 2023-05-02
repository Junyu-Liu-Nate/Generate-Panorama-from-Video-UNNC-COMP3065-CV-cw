import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from helpers import detect_content_range, fill_outermost_pixels
from perspective import postprocess

from deblur import blur_edge, motion_kernel, defocus_kernel, estimate_motion

def deblur(frame_list, magnitude, angle):
    deblur_frame_list = []
    for i in range(len(frame_list) - 1):
        print(f'Deblur for frame {i} in {len(frame_list)}.')
        frame1 = frame_list[i]
        frame2 = frame_list[i + 1]
        
        frame1_r, frame1_g, frame1_b = cv2.split(frame1)
        frame1_list = [frame1_r, frame1_g, frame1_b]
        frame2_r, frame2_g, frame2_b = cv2.split(frame2)
        frame2_list = [frame2_r, frame2_g, frame2_b]
        
        ang_list = []
        d_list = []

        for j in range(3):
            ang, d = estimate_motion([frame1_list[j], frame2_list[j]])
            ang_list.append(ang)
            d_list.append(d)
        
        ang_avg = int(np.mean(ang_list))
        d_avg = int(np.mean(d_list))
        # print(ang_avg, d_avg)

        # if d_avg == 0:
        #     deblur_frame_list.append(frame1)
        #     continue

        #----- Use psf and Wiener deconvolution for each channel of the frame-----#
        img = frame_list[i]
        b, g, r = cv2.split(img)
        img_list = [b, g, r]
        res_list = []
        for i in range(3):
            img = img_list[i]
            img = blur_edge(img)
            IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

            # ang = 0
            # d = 55
            ang_avg = angle
            d_avg = magnitude
            noise = 10**(-0.1*25)

            # psf = motion_kernel(ang, d)
            psf = motion_kernel(ang_avg, d_avg)

            psf /= psf.sum()
            psf_pad = np.zeros_like(img)
            kh, kw = psf.shape
            psf_pad[:kh, :kw] = psf
            PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
            PSF2 = (PSF**2).sum(-1)
            iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
            RES = cv2.mulSpectrums(IMG, iPSF, 0)
            res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
            res = np.roll(res, -kh//2, 0)
            res = np.roll(res, -kw//2, 1)

            # res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

            res_list.append(res)
        res = cv2.merge(res_list)
        res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        deblur_frame_list.append(res)

    return deblur_frame_list

def panorama_deconv(test_name, offset_x, offset_y, if_write_frame, magnitude, angle):
    vid_path = '../vid/' + test_name + '.mp4'
    left_img_path = '../img/' + test_name + '_left.jpg'
    right_img_path = '../img/' + test_name + '_right.jpg'
    final_img_path = '../img/' + test_name + '_final.jpg'

    ### Open the video file
    cap = cv2.VideoCapture(vid_path)
    ### Get the total number of frames in the video
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total number of frames in the video: {frame_count}')

    mag = int(magnitude)
    ang = int(angle)

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

    #----- Deblur frames -----#
    right_frames = deblur(right_frames, mag, ang)
    left_frames = deblur(left_frames, mag, ang)

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
    init_pano = panorama_right

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

        # for x in range(50):
        #     panorama_right = fill_outermost_pixels(panorama_right)

        # w, h = detect_content_range(prev_frame)
        w, h = detect_content_range(init_pano)

        ### Add the previoys frame at the middle left and blend it with the current frame
        mask1 = (panorama_right[:HEIGHT,:w] == 0).all(axis=2) & (init_pano[:HEIGHT,:w] != 0).any(axis=2)
        panorama_right[:HEIGHT,:w][mask1] = init_pano[:HEIGHT,:w][mask1]
        # mask1 = (panorama_right[:HEIGHT,:w] == 0).all(axis=2) & (prev_frame[:HEIGHT,:w] != 0).any(axis=2)
        # panorama_right[:HEIGHT,:w][mask1] = prev_frame[:HEIGHT,:w][mask1]
        # mask2 = (panorama_right[:HEIGHT,:w] != 0).any(axis=2) & (prev_frame[:HEIGHT,:w] != 0).any(axis=2)
        # panorama_right[:HEIGHT,:w][mask2] = prev_frame[:HEIGHT,:w][mask2] / 2 + panorama_right[:HEIGHT,:w][mask2] / 2

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

        # panorama_left = fill_outermost_pixels(panorama_left)
        for x in range(50):
            panorama_left = fill_outermost_pixels(panorama_left)

        w_, h_ = detect_content_range(prev_frame)
        h_ = h + offset_y - h_
        w_ = w + offset_x - w_

        ### Add the previoys frame at the middle right and blend it with the current frame
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

    # panorama_final = fill_outermost_pixels(panorama_final)
    # panorama_final = fill_outermost_pixels(panorama_final)
    # panorama_final = fill_outermost_pixels(panorama_final)
    for x in range(50):
        panorama_final = fill_outermost_pixels(panorama_final)

    mask1 = (panorama_final[:HEIGHT,:WIDTH] == 0).all(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    panorama_final[:HEIGHT,:WIDTH][mask1] = panorama_left[:HEIGHT,:WIDTH][mask1]
    mask2 = (panorama_final[:HEIGHT,:WIDTH] <= 50).any(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    panorama_final[:HEIGHT,:WIDTH][mask2] = panorama_left[:HEIGHT,:WIDTH][mask2]
    # mask2 = (panorama_final[:HEIGHT,:WIDTH] != 0).any(axis=2) & (panorama_left[:HEIGHT,:WIDTH] >= 150).any(axis=2)
    # panorama_final[:HEIGHT,:WIDTH][mask2] = panorama_left[:HEIGHT,:WIDTH][mask2] / 2 + panorama_final[:HEIGHT,:WIDTH][mask2] / 2

    cv2.imwrite(final_img_path, panorama_final)


# ### Choices for test_name: ['test_video_1', 'test_video_2', 'test_video_3'] ### 
# test_name = 'test_video_5blur'

# h, w = [1080, 1920]
# offset_x = int(w*1.8)
# offset_y = int(h*0.5)

# ### Flag to determine whether whether write frame-by-frame results when stitching
# if_write_frame = True

# panorama_deconv(test_name, offset_x, offset_y, if_write_frame)
# postprocess(test_name)