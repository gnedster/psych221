import cv2
import numpy as np
from matplotlib import pyplot as plt

nearest = cv2.Canny(cv2.imread('images/asianwoman_nearest_ip.png',0), 100, 200)
cnn = cv2.Canny(cv2.imread('images/asianwoman_cnn_ip.png',0), 100, 200)
high = cv2.Canny(cv2.imread('images/asianwoman_high_ip.png',0), 100, 200)
bilinear = cv2.Canny(cv2.imread('images/asianwoman_bilinear_ip.png',0), 100, 200)

plt.subplot(221),plt.imshow(nearest,cmap = 'gray')
plt.title('Nearest Neighbor'), plt.xticks([]), plt.yticks([])
plt.subplot(222),plt.imshow(cnn,cmap = 'gray')
plt.title('CNN'), plt.xticks([]), plt.yticks([])
plt.subplot(223),plt.imshow(bilinear,cmap = 'gray')
plt.title('Bilinear Interpolation'), plt.xticks([]), plt.yticks([])
plt.subplot(224),plt.imshow(high,cmap = 'gray')
plt.title('Simulation'), plt.xticks([]), plt.yticks([])
plt.show()