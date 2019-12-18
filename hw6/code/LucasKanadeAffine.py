import numpy as np
from scipy.interpolate import RectBivariateSpline
import cv2
from scipy.ndimage import affine_transform
import matplotlib.pyplot as pl


def LucasKanadeAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image
	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]
    # put your implementation here

    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    length = 6

    p = np.zeros ((length,1))

    dim = It.shape[0] * It.shape[1]

    grad_x, grad_y = np.gradient(It)


    rows_it=  range (0, It.shape[0])

    cols_it = range(0, It.shape[1])

    rows_it1=  range (0, It1.shape[0])
    cols_it1 = range(0, It1.shape[1])

    row, col = np.meshgrid(rows_it, cols_it)



    # pts = np.vstack(((row.reshape(1, -1)), (col.reshape(1, -1)), mask))
    #
    # spline_it = RectBivariateSpline(rows_it, cols_it, It)
    #
    # spline_it1 = RectBivariateSpline(rows_it1, cols_it1, It1)


    threshold = 0.1

    delta =100

    w = It.shape[1]
    h =  It.shape[0]

    p1 = p[0, 0]
    p2 = p[1, 0]
    p3 = p[2, 0]
    p4 = p[3, 0]
    p5 = p[4, 0]
    p6 = p[5, 0]


    while (delta >= threshold):

        p_warped = np.array([[1 + p1, p3, p5], [p2, 1 + p4, p6], [0, 0, 1]])

        img_warped = affine_transform(It1,p_warped)

        mask=  img_warped > 0.05

        img_masked = mask * It

        mask_warped = mask* img_warped

        b = img_warped - mask_warped

        b = b.reshape(b.size, 1)

        x_grad, y_grad = np.gradient (It1)

        row = row. reshape (x_grad.size,1)

        col = col. reshape (x_grad.size, 1)

        y_grad = y_grad.reshape(x_grad.size, 1)

        x_grad = x_grad.reshape(x_grad.size, 1)


        M1 = y_grad*col

        M2 = x_grad*col

        M3 = y_grad*row

        M4 = x_grad* row

        M5 = x_grad.reshape(x_grad.size, 1)

        M6 = y_grad.reshape(x_grad.size, 1)


        A = np.concatenate((M1, M2, M3, M4, M5, M6), axis=1)

        H = np.dot(A.T, A)

        err = np.dot(A.T, b)

        solved = np.linalg.solve(H, err)

        delta = np.sum(solved*solved)

        p = p + solved

    return p




#
# frames = np.load('../data/aerialseq.npy')
# img0= frames[:,:,0]
# img1= frames[:,:,1]
# M = LucasKanadeAffine(img0,img1)
# print (M)