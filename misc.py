import cv2
import numpy as np

def cylindricalWarp(img):
    """This function returns the cylindrical warp for a given image and intrinsics matrix K"""
    h, w = img.shape[:2]
    K = np.array([[3773.33, 0, w / 2], [0, 3204.55, h / 2], [0, 0, 1]])

    h_, w_ = img.shape[:2]
    # pixel coordinates
    y_i, x_i = np.indices((h_, w_))
    X = np.stack([x_i, y_i, np.ones_like(x_i)], axis=-1).reshape(h_ * w_, 3)  # to homog
    Kinv = np.linalg.inv(K)
    X = Kinv.dot(X.T).T  # normalized coords
    # calculate cylindrical coords (sin\theta, h, cos\theta)
    A = np.stack([np.sin(X[:, 0]), X[:, 1], np.cos(X[:, 0])], axis=-1).reshape(w_ * h_, 3)
    B = K.dot(A.T).T  # project back to image-pixels plane
    # back from homog coords
    B = B[:, :-1] / B[:, [-1]]
    # make sure warp coords only within image bounds
    B[(B[:, 0] < 0) | (B[:, 0] >= w_) | (B[:, 1] < 0) | (B[:, 1] >= h_)] = -1
    B = B.reshape(h_, w_, -1)

    # img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)  # for transparent borders...
    # warp the image according to cylindrical coords
    return cv2.remap(img, B[:, :, 0].astype(np.float32), B[:, :, 1].astype(np.float32), cv2.INTER_AREA, borderMode=cv2.BORDER_TRANSPARENT)

image = cv2.imread('img/6.jpg')
warp_img = cylindricalWarp(image)
cv2.imwrite('img/test_warp.jpg', warp_img)