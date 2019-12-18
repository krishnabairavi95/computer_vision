'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import scipy.ndimage as ndimage
import helper
import matplotlib.pyplot as plt
import submission
from mpl_toolkits import mplot3d

if __name__ == "__main__":


    some_corresp = np.load('../data/some_corresp.npz')
    pts1 = some_corresp['pts1']
    pts2 = some_corresp['pts2']

    im1 = plt.imread('../data/im1.png')
    im2 = plt.imread('../data/im2.png')

    temple_coords = np.load('../data/templeCoords.npz')
    x1 = temple_coords['x1']
    y1 = temple_coords['y1']

    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']

    h= im1.shape[0]
    w=im1.shape[1]
    M = max(h, w)

    ## Using eight points:

    F = submission.eightpoint(pts1, pts2, M)
    E = submission.essentialMatrix(F, K1, K2)

    I= np.eye(3)

    zero_array= np.zeros((3, 1))
    iter_val=4
    M1 = np.concatenate((I, zero_array),axis=1)
    C1 = np.matmul(K1 ,M1)

    M2_prev = helper.camera2(E)

    for i in range(iter_val):
        C2 = np.matmul(K2 ,M2_prev[:, :, i])
        W, err = submission.triangulate(C1, pts1, C2, pts2)
        if np.all(W[:, 2] > 0):
            M2 = M2_prev[:, :, i]
            break

    x2 = []
    y2 = []
    for i in range(len(x1)):
        a, b = submission.epipolarCorrespondence(
            im1, im2, F, x1[i][0], y1[i][0])
        x2.append(a)
        y2.append(b)

   #pts1 = np.hstack((x1, y1))
    pts1 = np.concatenate((x1, y1), axis=1)
    pts2 = np.vstack((x2, y2))
    pts2= np.transpose(pts2)


    P, temp = submission.triangulate(C1, pts1, C2, pts2)

    # np.savez('q4_1.npz', F=F, pts1=pts1, pts2=pts2)
    # np.savez('q4_2.npz', F=F, M1=M1, M2=M2, C1=C1, C2=C2)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(P[:, 0], P[:, 1], P[:, 2])
    plt.show()