import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade



# write your script here, we recommend the above libraries for making your animation


input_data =  np.load("./../data/carseq.npy")

_,_,frames = input_data.shape

rect0 = [59, 116, 145, 151]
rect1 = [59, 116, 145, 151]
rect2 = [59, 116, 145, 151]


rect = np.array([59, 116, 145, 151])
rect_first = np.array([59, 116, 145, 151])
rect_second = np.array ([59, 116, 145, 151])


template=input_data[:,:,0]

p_temp = np.zeros (2)

for frame in range (1, frames):

    curr_img= input_data[:, :, frame - 1]

    nxt_img = input_data[:, :, frame]

    p_temp = LucasKanade(template, curr_img, rect)

    p_no_temp = LucasKanade(curr_img, nxt_img,rect)

    rect1[0] = rect1[0] + p_no_temp[1]

    rect2[0] = rect0[0] + p_temp[1]

    rect1[1] = rect1[1] + p_no_temp[0]

    rect2[1] = rect0[1] + p_temp[0]

    rect1[2] = rect1[2] + p_no_temp[1]

    rect2[2] = rect0[2] + p_temp[1]

    rect1[3] = rect1[3] + p_no_temp[1]

    rect2[3] = rect0[3] + p_temp[0]

    rect1 = np.asarray(rect1)

    rect_first= np.vstack ((rect_first, rect1))

    rect2 = np.asarray(rect2)

    rect_second = np.vstack((rect_second, rect2))


    if ((frame - 1) == 0 or (frame - 1) == 99 or (frame - 1) == 199 or (frame - 1) == 299 or (frame - 1) == 399):

        fig = plt.figure()

        plt.imshow(curr_img, cmap='gray')


        ## First patch

        col_range = rect1[3] - rect[1]

        row_range = rect1[2] - rect1[0]

        first = patches.Rectangle((rect1[0], rect1[1]), row_range, col_range, edgecolor="r",linewidth=1)


        ## Second Patch

        col_range = rect2[3] - rect2[1]

        row_range = rect2[2] - rect2[0]

        second= patches.Rectangle((rect2[0], rect2[1]), row_range, col_range, edgecolor="b", linewidth=1)

        ax = plt.gca()

        ax.add_patch(first)

        ax.add_patch(second)

        plt.show ()

    np.save("carseqrects-wcrt.npy", rect)