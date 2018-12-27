import numpy as np
import cv2 #Python 3 prebuild: https://pypi.org/project/opencv-python/
import matplotlib.pyplot as plt

import bin.module.util as util
from bin.setting import path


"test"
#OpenCV follows BGR while Matplotlib RGB
img_bgr = cv2.imread(path.dataLake.imageFolder + 'img-1000/0-0.jpg')
b, g, r = cv2.split(img_bgr)
img = cv2.merge([r, g, b])
plt.imshow(img)
plt.show()
plt.close()
img.shape


#Contour detection (to remove the peripheral section)
#https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
#https://docs.opencv.org/3.3.1/d4/d73/tutorial_py_contours_begin.html
#https://www.quora.com/How-can-I-detect-an-object-from-static-image-and-crop-it-from-the-image-using-openCV
#Gray scale
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

#Blur image
gray = cv2.blur(src_gray, 3)

edged = cv2.Canny(gray, 10, 250)
edged.shape
plt.imshow(edged, cmap='gray')
plt.show()
plt.close()
 
#applying closing function 
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
plt.imshow(closed, cmap='gray')
plt.show()
plt.close()


#Adding crop image
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
plt.imshow(cnts[1], cmap='gray')
plt.show()
plt.close()

idx = 0
for c in cnts:
	x, y, w, h = cv2.boundingRect(c)
	if w > 50 and h > 50:
		idx += 1
		new_img = image[y : y + h, x : x + w]
		cv2.imwrite(str(idx) + '.png', new_img)
