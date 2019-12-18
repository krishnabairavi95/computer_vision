
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.patches as patches

# write your script here, we recommend the above libraries for making your animation
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import InverseCompositionAffine
from SubtractDominantMotion import SubtractDominantMotion


frames = np.load('../data/aerialseq.npy')

tot_frames = frames.shape[2]

fig = plt.figure()

for f in range(0, tot_frames-1):

    mask = SubtractDominantMotion(frames[:,:,f], frames[:,:,f+1])

    img = np.copy(frames[:,:,f+1])

    img = np.stack((img, img, img), axis=2) * 255.0

    img[:,:,2] += (mask.astype(np.float32)) * 100.0

    img = np.clip(img, 0, 255).astype(np.uint8)


    plt.imshow(img)

    if f in [30, 60, 90, 120]:

        #plt.savefig("frame"+str(f)+".png")
        plt.savefig("inv_frame"+str(f)+".png")
        # plt.pause(0.02)

