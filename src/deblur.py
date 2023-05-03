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
