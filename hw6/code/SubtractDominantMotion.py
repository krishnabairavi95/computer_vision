import numpy as np
from scipy.ndimage.morphology import binary_erosion
from LucasKanadeAffine import LucasKanadeAffine
from InverseCompositionAffine import *
from scipy.ndimage import affine_transform
from scipy.ndimage import binary_dilation
from scipy.ndimage import generate_binary_structure
import matplotlib.pyplot as plt


def SubtractDominantMotion(image1, image2):
	# Input:
	#	Images at time t and t+1
	# Output:
	#	mask: [nxm]
    # put your implementation here

    mask = np.ones(image1.shape, dtype=bool)

    struct1 = generate_binary_structure(2, 2)

   # p = LucasKanadeAffine(image1, image2)

    p = InverseCompositionAffine(image1, image2)

    p1 = p[0, 0]
    p2 = p[1, 0]
    p3 = p[2, 0]
    p4 = p[3, 0]
    p5 = p[4, 0]
    p6 = p[5, 0]

    p_warp = np.array([[1 + p1, p3, p5], [p2, 1 + p4, p6], [0, 0, 1]])

    warpedImage = affine_transform(image2, p_warp)

    plt.imshow (warpedImage)

    plt.show()

    b = image1 - warpedImage

    b = b > 0.10

    mask = binary_dilation(b, structure=struct1)

    mask = binary_erosion(mask, structure=struct1)

    return mask

