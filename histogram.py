import cv2
import numpy as np
from os import listdir
from matplotlib import pyplot as plt

color = ('b','g','r')

if __name__ == '__main__':
  files = listdir('images')

  for i,col in enumerate(color):
    histr = np.zeros([256, 1])
    for file in files:
      if 'cnn_ip' in file:
        img = cv2.imread('images/' + file)
        histr = np.add(cv2.calcHist([img],[i],None,[256],[0,256]), histr)

    plt.plot(histr,color = col)
    plt.xlim([0,256])

  plt.show()