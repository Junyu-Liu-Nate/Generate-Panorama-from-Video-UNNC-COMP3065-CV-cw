""" 
    This code loads a motion-blurred image and defines a point spread function (PSF) for motion blur from left to right. 
    Then it defines a signal-to-noise ratio (SNR) and deconvolves the image using Wiener deconvolution. 
    Finally it shows the deblurred image.
"""

import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.signal import fftconvolve
from scipy.signal import wiener
from numpy.fft import fft2, ifft2
# from skimage.restoration import richardson_lucy
from astropy.convolution import convolve

# def wiener_deconvolution(image, psf, snr):
#     # Compute the OTF of the PSF
#     otf = np.fft.fft2(psf)

#     # Compute the Wiener filter
#     wiener_filter = np.conj(otf) / (np.abs(otf) ** 2 + 1 / snr)

#     # Deconvolve the image using the Wiener filter
#     deblurred_image = np.real(np.fft.ifft2(np.fft.fft2(image) * wiener_filter))

#     return deblurred_image

def wiener_deconvolution(blurred_image, psf, snr):
    # Compute the Fourier transforms of the image and the PSF
    image_fft = fft2(blurred_image)
    psf_fft = fft2(psf, s=blurred_image.shape)

    # Compute the Wiener filter
    wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + 1 / snr)

    # Apply the Wiener filter to the image
    deconvolved_img = np.real(ifft2(wiener_filter * image_fft))

    return deconvolved_img

    # # Compute the Wiener filter
    # wiener_filter = wiener(blurred_image, psf.shape, noise=snr)
    
    # # Compute the 2D Fourier transform of the blurred image and the Wiener filter
    # blurred_image_fft = fft2(blurred_image)
    # wiener_filter_fft = fft2(wiener_filter, s=blurred_image.shape[:2])
    
    # # Multiply the two arrays element-wise in the frequency domain
    # deblurred_image_fft = blurred_image_fft * wiener_filter_fft
    
    # # Compute the inverse 2D Fourier transform of the result
    # deblurred_image = np.real(ifft2(deblurred_image_fft))
    
    # return deblurred_image

    # # Compute the Wiener filter
    # wiener_filter = wiener(blurred_image, psf.shape, noise=snr)
    
    # # Apply the Wiener filter to the blurred image
    # deblurred_image = convolve2d(blurred_image, wiener_filter, mode='same')
    
    # return deblurred_image

def estimate_psf(blurred_image, neighboring_frames):
    ### Compute the average of the neighboring frames
    avg_frame = np.mean(neighboring_frames, axis=0)
    
    ### Subtract the average frame from the blurred image
    diff = blurred_image - avg_frame

    # ### Create the kernel by flipping the difference image
    # kernel = diff[::-1, ::-1]
    
    # ### Pad the kernel with zeros if necessary to make its size odd
    # if kernel.shape[0] % 2 == 0:
    #     kernel = np.pad(kernel, ((1, 0), (0, 0)), mode='constant')
    # if kernel.shape[1] % 2 == 0:
    #     kernel = np.pad(kernel, ((0, 0), (1, 0)), mode='constant')

    # ### Manually Normalize the kernel
    # kernel = kernel / kernel.sum()

    ### Estimate the PSF by computing the autocorrelation of the difference image
    psf = convolve2d(diff, diff[::-1, ::-1], mode='same')
    # psf = fftconvolve(diff, diff[::-1, ::-1], mode='same')
    # psf = convolve(diff, kernel, boundary='fill', fill_value=0)
    
    ### Normalize the PSF
    psf = psf / psf.sum()
    
    return psf

# def estimate_psf(blurred_image, num_iterations=10):
#     # Initialize the PSF with a small random value
#     psf = np.ones((5, 5)) / 25
    
#     # Use the Richardson-Lucy deconvolution algorithm to estimate the PSF
#     for i in range(num_iterations):
#         deconvolved = richardson_lucy(blurred_image, psf)
#         psf = richardson_lucy(blurred_image, deconvolved)
    
#     return psf

test_name = 'video_blur'
vid_path = 'vid/' + test_name + '.mp4'
output_img_path = 'img/blur/' + test_name + '_blur.jpg'
output_deblur_img_path = 'img/blur/' + test_name + '_deblur.jpg'

# # Load the motion-blurred image
# image = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
# Load the video
cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
    raise Exception('Could not open video')
# Read the first, 2, 3 frame from the video
ret, image1 = cap.read()
image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
ret, image2 = cap.read()
image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
ret, image3 = cap.read()
image3_gray = cv2.cvtColor(image3, cv2.COLOR_RGB2GRAY)

new_size = (192, 108)
image1_gray = cv2.resize(image1_gray, new_size)
image2_gray = cv2.resize(image2_gray, new_size)
image3_gray = cv2.resize(image3_gray, new_size)

### Define the point spread function (PSF) for motion blur from left to right
# print(f'image2_gray: {image2_gray}')
psf = estimate_psf(image2_gray.copy(), np.array([image1_gray, image3_gray]))
# print(f'psf shape is {psf.shape}')

### Define the signal-to-noise ratio (SNR)
snr = 1e-2

### Deconvolve the image using Wiener deconvolution
deblurred_image = wiener_deconvolution(image2_gray, psf, snr)

deblurred_image = cv2.normalize(deblurred_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
# print(deblurred_image)

# Show the deblurred image
# cv2.imshow('Deblurred Image', deblurred_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
cv2.imwrite(output_img_path, image2_gray)
cv2.imwrite(output_deblur_img_path, deblurred_image)