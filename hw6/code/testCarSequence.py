import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches
from LucasKanade import LucasKanade


# write your script here, we recommend the above libraries for making your animation

input_data =  np.load("./../data/carseq.npy")

frames = input_data.shape[2]


coords = [59, 116, 145, 151]

boxes = coords

boxes = np.asarray(boxes)


for  frame in range (1, frames):

    curr_img= input_data[:, :, frame - 1]

    nxt_img = input_data[:, :, frame]

    p = LucasKanade(curr_img, nxt_img, coords)

    # px= p[1]
    #

    # py= p[0]

    coords = np.array ([coords[0] + p[1], coords[1] + p[0], coords[2] + p[1], coords[3] + p[0]])

    boxes = np.vstack ((boxes, coords))


    if ((frame-1) ==0 or (frame-1) ==99 or (frame-1) ==199 or (frame-1) ==299 or (frame-1) ==399):

        fig = plt. figure ()

        plt. imshow (curr_img, cmap ='gray')


        col_range = coords [3] - coords [1]

        row_range = coords[2] - coords [0]

        patch = patches.Rectangle((coords[0], coords[1]), row_range, col_range, edgecolor="r", facecolor="none", linewidth=3)

        ax = plt.gca()

        ax.add_patch(patch)

        plt. show ()


np.save("carseqrects.npy", boxes)