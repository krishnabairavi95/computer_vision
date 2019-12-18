import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from alignChannels import alignChannels
import cv2
# Problem 1: Image Alignment


# 1. Load images (all 3 channels)
red = np.load('../data/red.npy')
green = np.load('../data/green.npy')
blue = np.load('../data/blue.npy')


#plt.imshow(green)
#plt.show()

# 2. Find best alignment
rgbResult = alignChannels(red, green, blue)
cv2.imwrite('../results/rgb_output.jpg', rgbResult)

#plt.imshow(rgbResult)
#plt.show()
# 3. save result to rgb_output.jpg (IN THE "results" FOLDER)

