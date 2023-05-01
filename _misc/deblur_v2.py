import cv2
import numpy as np
from scipy.signal import convolve2d, wiener
from numpy.fft import fft2, ifft2

def wiener_deconvolution(image, psf, snr):
    # Compute the Fourier transforms of the image and the PSF
    image_fft = fft2(image)
    psf_fft = fft2(psf, s=image.shape)

    # Compute the Wiener filter
    wiener_filter = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + 1 / snr)

    # Apply the Wiener filter to the image
    deconvolved_img = np.real(ifft2(wiener_filter * image_fft))

    return deconvolved_img

test_name = 'video_blur'
vid_path = 'vid/' + test_name + '.mp4'
output_img_path = 'img/blur/' + test_name + '_ori.jpg'
output_blur_img_path = 'img/blur/' + test_name + '_blur.jpg'
output_deblur_img_path = 'img/blur/' + test_name + '_deblur.jpg'

cap = cv2.VideoCapture(vid_path)
if not cap.isOpened():
    raise Exception('Could not open video')
ret, image1 = cap.read()
image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)

# psf = np.ones((5, 5)) / 25
psf = np.zeros((1, 50))
psf[0, :] = 1 / 50
image1_blur = convolve2d(image1_gray, psf, 'same')

deconvolved_img = wiener_deconvolution(image1_blur, psf, snr=1e-1)

# Rescale the pixel values to the range [0, 255]
deconvolved_img = cv2.normalize(deconvolved_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite(output_img_path, image1_gray)
cv2.imwrite(output_blur_img_path, image1_blur)
cv2.imwrite(output_deblur_img_path, deconvolved_img)
