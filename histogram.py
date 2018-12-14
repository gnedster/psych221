import cv2
import numpy as np
from os import listdir
from matplotlib import pyplot as plt

color = ('b','g','r')

if __name__ == '__main__':
  files = listdir('images')

  for i,col in enumerate(color):
    nearest = np.zeros([256, 1])
    cnn = np.zeros([256, 1])
    bilinear = np.zeros([256, 1])
    high = np.zeros([256, 1])
    for file in files:
      img = cv2.imread('images/' + file)
      if 'nearest_ip' in file:
        nearest = np.add(cv2.calcHist([img],[i],None,[256],[0,256]), nearest)
      elif 'cnn_ip' in file:
        cnn = np.add(cv2.calcHist([img],[i],None,[256],[0,256]), cnn)
      elif 'bilinear_ip' in file:
        bilinear = np.add(cv2.calcHist([img],[i],None,[256],[0,256]), bilinear)
      elif 'high_ip' in file:
        high = np.add(cv2.calcHist([img],[i],None,[256],[0,256]), high)

    plt.subplot(221), plt.plot(np.divide(nearest, len(files)),color = col), plt.xlim([0,256])
    plt.title('Nearest Neighbor')
    plt.subplot(222), plt.plot(np.divide(cnn, len(files)),color = col), plt.xlim([0,256])
    plt.title('CNN')
    plt.subplot(223), plt.plot(np.divide(bilinear, len(files)),color = col), plt.xlim([0,256])
    plt.title('Bilinear Interpolation')
    plt.subplot(224), plt.plot(np.divide(high, len(files)),color = col), plt.xlim([0,256])
    plt.title('Simulation')

  plt.show()