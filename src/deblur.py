import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2

def blur_edge(img, d=31):
    h, w  = img.shape[:2]
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, (2*d+1, 2*d+1), -1)[d:-d,d:-d]
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    w = np.minimum(np.float32(dist)/d, 1.0)
    return img*w + img_blur*(1-w)

def motion_kernel(angle, d, sz=65):
    if d <= 0:
        return np.zeros((sz, sz))
    
    kern = np.ones((1, d), np.float32)
    c, s = np.cos(angle), np.sin(angle)
    A = np.float32([[c, -s, 0], [s, c, 0]])
    sz2 = sz // 2
    A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((d-1)*0.5, 0))
    kern = cv2.warpAffine(kern, A, (sz, sz), flags=cv2.INTER_CUBIC)
    return kern

def defocus_kernel(d, sz=65):
    kern = np.zeros((sz, sz), np.uint8)
    cv2.circle(kern, (sz, sz), d, 255, -1, cv2.LINE_AA, shift=1)
    kern = np.float32(kern) / 255.0
    return kern

def richardson_lucy(image, psf, iterations=50):
    """
    Deconvolve an image using the Richardson-Lucy deconvolution algorithm.

    :param image: The input image to be deconvolved.
    :param psf: The point spread function (PSF) of the blur.
    :param iterations: The number of iterations to perform.
    :return: The deconvolved image.
    """
    # Initialize the output image
    out = np.full(image.shape, 0.5)

    # Pad the PSF with zeros to the same size as the image
    psf_padded = np.zeros_like(image)
    kh, kw = psf.shape
    psf_padded[:kh, :kw] = psf

    # Compute the FFT of the PSF
    psf_fft = fft2(psf_padded)

    # Perform the Richardson-Lucy deconvolution
    for i in range(iterations):
        # Compute the estimated blurred image
        out_fft = fft2(out)
        est_blur_fft = out_fft * psf_fft
        est_blur = np.real(ifft2(est_blur_fft))

        # Compute the relative blur
        relative_blur = image / est_blur

        # Compute the correction factor
        relative_blur_fft = fft2(relative_blur)
        correction_fft = relative_blur_fft * psf_fft.conj()
        correction = np.real(ifft2(correction_fft))

        # Update the output image
        out *= correction

    out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return out

def estimate_motion(frames):
    """
    Estimate the motion between a sequence of frames using optical flow.

    :param frames: A list of consecutive frames from a video.
    :return: The estimated angle and displacement of the motion.
    """
    #----- Use optical flow to estimate psf -----#
    # # Initialize variables to accumulate the motion estimates
    # avg_flow_x = 0
    # avg_flow_y = 0

    # # Loop over pairs of consecutive frames
    # for i in range(len(frames) - 1):
    #     # Calculate the optical flow between the two frames
    #     flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)

    #     # Accumulate the motion estimates
    #     avg_flow_x += np.mean(flow[...,0])
    #     avg_flow_y += np.mean(flow[...,1])

    #----- Use feature matching to estimate psf -----#
    # Initialize variables to accumulate the motion estimates
    avg_flow_x = 0
    avg_flow_y = 0

    # Create an ORB feature detector and a brute-force matcher
    orb = cv2.ORB_create()
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Loop over pairs of consecutive frames
    for i in range(len(frames) - 1):
        # Detect features in the first frame using the ORB feature detector
        kp1, des1 = orb.detectAndCompute(frames[i], None)
        kp2, des2 = orb.detectAndCompute(frames[i+1], None)

        # Match the features in the first frame with those in the second frame using a brute-force matcher
        matches = bf.match(des1, des2)

        # Accumulate the motion estimates
        avg_flow_x += np.mean([kp2[match.trainIdx].pt[0] - kp1[match.queryIdx].pt[0] for match in matches])
        avg_flow_y += np.mean([kp2[match.trainIdx].pt[1] - kp1[match.queryIdx].pt[1] for match in matches])

    # Average the motion estimates
    avg_flow_x /= len(frames) - 1
    avg_flow_y /= len(frames) - 1

    # Calculate the angle and displacement d for the motion_kernel function
    ang = np.arctan2(avg_flow_y, avg_flow_x)
    d = int(np.sqrt(avg_flow_x**2 + avg_flow_y**2))

    return ang, d

# test_name = 'test_video_1_motionblur_10'
# vid_path = 'vid/' + test_name + '.mp4'
# output_img_path = 'img/blur/' + test_name + '_blur.jpg'
# output_deblur_img_path = 'img/blur/' + test_name + '_deblur.jpg'

# # Load the video
# cap = cv2.VideoCapture(vid_path)
# if not cap.isOpened():
#     raise Exception('Could not open video')
# # Read the first, 2, 3 frame from the video
# ret, image1 = cap.read()
# # image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
# ret, image2 = cap.read()
# # image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
# ret, image3 = cap.read()
# # image3_gray = cv2.cvtColor(image3, cv2.COLOR_RGB2GRAY)

# frame1 = image1
# frame2 = image2
# frame3 = image3

# frame1_r, frame1_g, frame1_b = cv2.split(frame1)
# frame1_list = [frame1_r, frame1_g, frame1_b]
# frame2_r, frame2_g, frame2_b = cv2.split(frame2)
# frame2_list = [frame2_r, frame2_g, frame2_b]
# frame3_r, frame3_g, frame3_b = cv2.split(frame3)
# frame3_list = [frame3_r, frame3_g, frame3_b]
# # psf_list = []
# ang_list = []
# d_list = []
# for i in range(3):
#     #----- Use optical flow to estimate psf -----#
#     # # Calculate the optical flow between the two frames
#     # flow = cv2.calcOpticalFlowFarneback(frame1_list[i], frame2_list[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)

#     # # Calculate the average motion in x and y direction
#     # avg_flow_x = np.mean(flow[...,0])
#     # avg_flow_y = np.mean(flow[...,1])
#     ang, d = estimate_motion([frame1_list[i], frame2_list[i], frame3_list[i]])

#     #----- Use feature matching to estimate psf -----#
#     # # Detect features in the first frame using the ORB feature detector
#     # orb = cv2.ORB_create()
#     # kp1, des1 = orb.detectAndCompute(frame1, None)
#     # kp2, des2 = orb.detectAndCompute(frame2, None)

#     # # Match the features in the first frame with those in the second frame using a brute-force matcher
#     # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     # matches = bf.match(des1, des2)

#     # # Calculate the average motion in x and y direction
#     # avg_flow_x = np.mean([kp2[match.trainIdx].pt[0] - kp1[match.queryIdx].pt[0] for match in matches])
#     # avg_flow_y = np.mean([kp2[match.trainIdx].pt[1] - kp1[match.queryIdx].pt[1] for match in matches])

#     # # Calculate the angle and displacement d for the motion_kernel function
#     # ang = np.arctan2(avg_flow_y, avg_flow_x)
#     # d = int(np.sqrt(avg_flow_x**2 + avg_flow_y**2))
#     print(f'Estimated for channel {i}: ang={ang}, d={d}')

#     # # d=int(d/2)
#     # noise = 10**(-0.1*25)
#     # psf = motion_kernel(ang, d)
    
#     # psf_list.append(psf)
#     ang_list.append(ang)
#     d_list.append(d)
# ang_avg = int(np.mean(ang_list))
# d_avg = int(np.mean(d_list))
# print(f'Averaged estimation: ang={ang_avg}, d={d_avg}')

# #----- Use optical flow to estimate psf -----#
# # # Calculate the optical flow between the two frames
# # flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

# # # Calculate the average motion in x and y direction
# # avg_flow_x = np.mean(flow[...,0])
# # avg_flow_y = np.mean(flow[...,1])

# #----- Use feature matching to estimate psf -----#
# # # Detect features in the first frame using the ORB feature detector
# # orb = cv2.ORB_create()
# # kp1, des1 = orb.detectAndCompute(frame1, None)
# # kp2, des2 = orb.detectAndCompute(frame2, None)

# # # Match the features in the first frame with those in the second frame using a brute-force matcher
# # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# # matches = bf.match(des1, des2)

# # # Calculate the average motion in x and y direction
# # avg_flow_x = np.mean([kp2[match.trainIdx].pt[0] - kp1[match.queryIdx].pt[0] for match in matches])
# # avg_flow_y = np.mean([kp2[match.trainIdx].pt[1] - kp1[match.queryIdx].pt[1] for match in matches])

# # Calculate the angle and displacement d for the motion_kernel function
# # ang = np.arctan2(avg_flow_y, avg_flow_x)
# # d = int(np.sqrt(avg_flow_x**2 + avg_flow_y**2))
# # print(f'Estimated ang={ang}, d={d}')

# img = image1
# cv2.imwrite(output_img_path, img)

# #----- Use psf and Wiener deconvolution for each channel of the frame-----#
# b, g, r = cv2.split(img)
# img_list = [b, g, r]
# res_list = []
# for i in range(3):
#     img = img_list[i]
#     img = blur_edge(img)
#     IMG = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)

#     ang = 0
#     d = 10
#     noise = 10**(-0.1*25)

#     # psf = motion_kernel(ang, d)
#     psf = motion_kernel(ang_avg, d_avg)
#     # psf = psf_list[i]

#     psf /= psf.sum()
#     psf_pad = np.zeros_like(img)
#     kh, kw = psf.shape
#     psf_pad[:kh, :kw] = psf
#     PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = kh)
#     PSF2 = (PSF**2).sum(-1)
#     iPSF = PSF / (PSF2 + noise)[...,np.newaxis]
#     RES = cv2.mulSpectrums(IMG, iPSF, 0)
#     res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
#     res = np.roll(res, -kh//2, 0)
#     res = np.roll(res, -kw//2, 1)

#     # res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

#     res_list.append(res)

# res = cv2.merge(res_list)
# res = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# cv2.imwrite('img/blur/' + test_name + '_deblur.jpg', res)

# # psf = motion_kernel(ang, d)
# # psf /= psf.sum()

# # # Split the image into its color channels
# # b, g, r = cv2.split(img)

# # # Deblur each color channel using the Richardson-Lucy algorithm
# # b_deblurred = richardson_lucy(b, psf)
# # g_deblurred = richardson_lucy(g, psf)
# # r_deblurred = richardson_lucy(r, psf)

# # # Merge the deblurred color channels back into a single image
# # deblurred = cv2.merge([b_deblurred, g_deblurred, r_deblurred])

# # # Save the deblurred image
# # cv2.imwrite('img/blur/' + test_name + '_deblur.jpg', deblurred)