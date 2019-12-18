'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import scipy.ndimage as ndimage
import helper
import submission
import cv2


if __name__ == "__main__":

    some_corresp = np.load('../data/some_corresp.npz')

    pts1 = some_corresp['pts1']
    pts2 = some_corresp['pts2']

    intrinsics = np.load('../data/intrinsics.npz')

    K1 = intrinsics['K1']
    K2 = intrinsics['K2']



    im1 = cv2.imread('../data/im1.png')
    im2 = cv2.imread('../data/im2.png')

    temple_coords = np.load('../data/templeCoords.npz')
    x1 = temple_coords['x1']
    y1 = temple_coords['y1']

    M = max(im1.shape[0],im1.shape[1])


    F_array = submission.eightpoint(pts1, pts2, M)
    E = submission.essentialMatrix(F_array, K1, K2)

    I= np.eye(3)

    iter_range=4
    zero_array=np.zeros((3, 1))
    M1 = np.hstack((I, zero_array))
    C1 = np.matmul(K1 ,M1)

    M2_prev = helper.camera2(E)

    for i in range(iter_range):
        C2 = np.matmul(K2 ,M2_prev[:, :, i])
        W, err = submission.triangulate(C1, pts1, C2, pts2)
        print (err)
        if np.all(W[:, 2] > 0):
            M2 = M2_prev[:, :, i]
            break

    np.savez('q3_3.npz', M2=M2, C2=C2, P=W)

