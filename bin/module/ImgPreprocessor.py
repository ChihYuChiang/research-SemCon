import numpy as np
import cv2 #Python 3 prebuild: https://pypi.org/project/opencv-python/
import matplotlib.pyplot as plt

import bin.module.util as util


#OpenCV follows BGR while Matplotlib RGB
img_bgr = cv2.imread('data/img/0-0.jpg')
b, g, r = cv2.split(img_bgr)
img = cv2.merge([r, g, b])
plt.imshow(img)
plt.show()
plt.close()
img.shape