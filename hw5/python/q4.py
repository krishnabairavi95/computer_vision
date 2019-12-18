# import numpy as np
#
# import skimage
# from skimage.measure import  regionprops,label
# from skimage.color import label2rgb, rgb2gray
# import  skimage.restoration
# from skimage.filters import gaussian
# from skimage.filters import threshold_otsu
# from  skimage.morphology import closing, square
# from  skimage.segmentation import clear_border
#
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
#
#
# # takes a color image
# # returns a list of bounding boxes and black_and_white image
# # insert processing in here
# # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes
# # this can be 10 to 15 lines of code using skimage functions
#
#
# def findLetters(image):
#
#     bboxes = []
#
#     bw = rgb2gray(image)
#
#     otsu_threshold = threshold_otsu(bw)
#
#     bw = closing(bw < otsu_threshold , square(10))
#
#     cleared = clear_border(bw)
#
#     label_image = label(cleared)
#
#     image_label_overlay = label2rgb(label_image, image=image)
#
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     ax.imshow(image_label_overlay)
#
#     for region in regionprops(label_image):
#
#         # take regions with large enough areas
#         if region.area >= 100:
#
#             # draw rectangle around segmented coins
#             minr, minc, maxr, maxc = region.bbox
#
#             rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
#                                       fill=False, edgecolor='red', linewidth=2)
#
#             ax.add_patch(rect)
#
#             bboxes.append(region.bbox)
#
#     # ax.set_axis_off()
#     # plt.tight_layout()
#     # plt.show()
#     # plt.imshow(bw)
#
#
#     return bboxes, bw
#
#
# #
# # if __name__== "__main__":
# #
# #     image= skimage.io.imread ('/Users/krishnabairavi/Desktop/Fall19/CV/hw5/images/04_deep.jpg')
# #     bboxes, bw =findLetters(image)
# #
#
#

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.color import rgb2gray
def findLetters(image):
    bboxes = []
    bw = rgb2gray(image)
    thresh = threshold_otsu(bw)
    bw = closing(bw < thresh, square(10))
    cleared = clear_border(bw)
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=image)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            bboxes.append(region.bbox)
    #ax.set_axis_off()
    #plt.tight_layout()
    #plt.show()
    #plt.imshow(bw)

    return bboxes, bw
