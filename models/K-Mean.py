import numpy as np
import cv2

img = cv2.imread('image.jpg')
img2 = img.reshape((-1, 3))
img2 = np.flotat32(img2)
criteria = (cv2.TERM_CRITERIA_EPA + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
k = 4
attempts = 10
ret, label, center = cv2.kmeans(img2, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.resshape(img.shape)
