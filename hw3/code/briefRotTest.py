import numpy as np
import cv2
import os
from keypointDetect import DoGdetector
import BRIEF
import matplotlib.pyplot as plt


img= cv2.imread('../data/model_chickenbroth.jpg')

height= img.shape[0]
width=img.shape[1]
image_center= (width / 2, height / 2)

locs,descs= BRIEF.briefLite(img)

angles= range(0,190,10)
print (angles)

total_matches=[]

for n in angles:
    rot_img= cv2.getRotationMatrix2D(image_center, n, 1)
    affine_img= cv2.warpAffine(img, rot_img, (height, width))
    locs1, desc1 = BRIEF.briefLite(affine_img)
    inter_matches = BRIEF.briefMatch(descs, desc1)
    total_matches.append(inter_matches.shape[0])


plt.bar(angles,total_matches,align='center')
plt.xlabel('Rotation angle (in degree)')
plt.ylabel('No of matches')
plt.title('Test with an image and its rotations')
plt.show()

