import numpy as np
import cv2
import skimage
import matplotlib.pyplot as plt
import time

from tests.skimage_polar_transform import polar2cart

# img = cv2.imread('images/retina.jpg')
# img = cv2.imread('images/LimbDark_marked.png')
img = cv2.imread('images/20140704_022325_4096_HMII.jpg')
# plt.imshow(img)
# plt.show()
# plt.clf()

start = time.time()
warped = polar2cart(img, (2057, 2051), preserve_range=True)
end = time.time()
print(end-start)

cv2.imwrite("tmp/skimage_pol_hmi.jpg", warped[:, :1870, :])





