import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from helpers import detect_content_range
from perspective import postprocess

def fill_outermost_pixels(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Create a mask for the content area
    mask = gray > 0
    
    # Find the contours of the content area
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the outermost pixels
    outermost_mask = np.zeros_like(gray)
    for contour in contours:
        cv2.drawContours(outermost_mask, [contour], -1, 1, 1)
    
    # Set the outermost pixels to pure black
    img[outermost_mask.astype(bool)] = 0
    
    return img

def single_image_inference(model, img):
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      img = img2tensor(img)
      # img = torch.from_numpy(img)
      # img = img.float()
      # Transpose the image data from HWC to CHW format
      # img = img.permute(2, 0, 1)
      model.feed_data(data={'lq': img.unsqueeze(dim=0)})

      if model.opt['val'].get('grids', False):
          model.grids()

      model.test()

      if model.opt['val'].get('grids', False):
          model.grids_inverse()

      visuals = model.get_current_visuals()
      sr_img = tensor2img([visuals['result']])

      # # Convert the numpy array to a torch tensor
      # sr_img_tensor = torch.from_numpy(sr_img)
      # # Transpose the image data from CHW to HWC format
      # sr_img_tensor = sr_img_tensor.permute(1, 2, 0)
      # # Convert the torch tensor back to a numpy array
      # sr_img = sr_img_tensor.numpy()

      return sr_img

def deblur(frame_list, model):
    deblur_frame_list = []
    current_frame = 0
    for frame in frame_list:
        current_frame += 1
        start_time = time.time()

        deblur_img = single_image_inference(model, frame)
        
        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f'Finish deblur for frame {current_frame} of {len(frame_list)}..........Process time: {elapsed_time:.2f} seconds')
        deblur_frame_list.append(deblur_img)

    # print(deblur_frame_list)
    return deblur_frame_list

def panorama_nafnet(test_name, offset_x, offset_y, if_write_frame, deblur_model):
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
    right_frames = deblur(right_frames, deblur_model)
    left_frames = deblur(left_frames, deblur_model)

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

        panorama_right = fill_outermost_pixels(panorama_right)

        w, h = detect_content_range(prev_frame)

        ### Add the previoys frame at the middle left and blend it with the current frame
        mask1 = (panorama_right[:HEIGHT,:w] == 0).all(axis=2) & (prev_frame[:HEIGHT,:w] != 0).any(axis=2)
        panorama_right[:HEIGHT,:w][mask1] = prev_frame[:HEIGHT,:w][mask1]
        mask2 = (panorama_right[:HEIGHT,:w] != 0).any(axis=2) & (prev_frame[:HEIGHT,:w] != 0).any(axis=2)
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
        print(f'Finish stitching right frame {current_frame} of {len(right_frames)}..........Process time: {elapsed_time:.2f} seconds')

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
        print(f'Finish stitching left frame {current_frame} of {len(left_frames)}..........Process time: {elapsed_time:.2f} seconds')

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

    panorama_final = fill_outermost_pixels(panorama_final)
    panorama_final = fill_outermost_pixels(panorama_final)
    panorama_final = fill_outermost_pixels(panorama_final)

    mask1 = (panorama_final[:HEIGHT,:WIDTH] == 0).all(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    panorama_final[:HEIGHT,:WIDTH][mask1] = panorama_left[:HEIGHT,:WIDTH][mask1]
    mask2 = (panorama_final[:HEIGHT,:WIDTH] <= 50).any(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    panorama_final[:HEIGHT,:WIDTH][mask2] = panorama_left[:HEIGHT,:WIDTH][mask2]
    # mask2 = (panorama_final[:HEIGHT,:WIDTH] != 0).any(axis=2) & (panorama_left[:HEIGHT,:WIDTH] != 0).any(axis=2)
    # panorama_final[:HEIGHT,:WIDTH][mask2] = panorama_left[:HEIGHT,:WIDTH][mask2] / 2 + panorama_final[:HEIGHT,:WIDTH][mask2] / 2

    cv2.imwrite(final_img_path, panorama_final)

def imread(img_path):
  img = cv2.imread(img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  return img
def img2tensor(img, bgr2rgb=False, float32=True):
    img = img.astype(np.float32) / 255.
    return _img2tensor(img, bgr2rgb=bgr2rgb, float32=float32)

# import torch
# import os

# # os.chdir('NAFNet/')
# from basicsr.models import create_model
# from basicsr.utils import img2tensor as _img2tensor, tensor2img, imwrite
# from basicsr.utils.options import parse

# opt_path = 'options/test/MyDataset/modified_NAFNet.yml'
# opt = parse(opt_path, is_train=False)
# opt['dist'] = False
# NAFNet = create_model(opt)
# os.chdir('../')

# ### Choices for test_name: ['test_video_1', 'test_video_2', 'test_video_3'] ### 
# test_name = 'video_blur'

# h, w = [1080, 1920]
# offset_x = int(w*1.8)
# offset_y = int(h*0.5)

# ### Flag to determine whether whether write frame-by-frame results when stitching
# if_write_frame = True

# panorama(test_name, offset_x, offset_y, if_write_frame, NAFNet)
# postprocess(test_name)