import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import affine_transform


def InverseCompositionAffine(It, It1):
	# Input:
	#	It: template image
	#	It1: Current image

	# Output:
	#	M: the Affine warp matrix [2x3 numpy array]

    # put your implementation here

    M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

    length = 6

   # p = np.zeros((length, 1))

    p = np.zeros((length,1))

    dim = It.shape[0] * It.shape[1]

    grad_x, grad_y = np.gradient(It)

    rows_it = range(0, It.shape[0])

    cols_it = range(0, It.shape[1])

    rows_it1 = range(0, It1.shape[0])

    cols_it1 = range(0, It1.shape[1])

    row, col = np.meshgrid(rows_it, cols_it)

    # pts = np.vstack(((row.reshape(1, -1)), (col.reshape(1, -1)), mask))
    #
    # spline_it = RectBivariateSpline(rows_it, cols_it, It)
    #
    # spline_it1 = RectBivariateSpline(rows_it1, cols_it1, It1)

    threshold = 1.6

    delta = 100

    w = It.shape[1]
    h = It.shape[0]


    p1 = p[0, 0]
    p2 = p[1, 0]
    p3 = p[2, 0]
    p4 = p[3, 0]
    p5 = p[4, 0]
    p6 = p[5, 0]
    #

    p_warped = np.array([[1 + p1, p3, p5], [p2, 1 + p4, p6], [0, 0, 1]])

    # mask = img_warped > 0.05
    #
    # img_masked = mask * It
    #
    # mask_warped = mask * img_warped

    x_grad, y_grad = np.gradient(It)

    row = row.reshape(x_grad.size, 1)

    col = col.reshape(x_grad.size, 1)

    y_grad = y_grad.reshape(x_grad.size, 1)

    x_grad = x_grad.reshape(x_grad.size, 1)

    M1 = y_grad * col

    M2 = x_grad * col

    M3 = y_grad * row

    M4 = x_grad * row

    M5 = y_grad.reshape(x_grad.size, 1)

    M6 = x_grad.reshape(x_grad.size, 1)

    A = np.hstack((M1, M2, M3, M4, M5, M6))

    H = np.dot(A.T, A)

    I_matrix = np.eye(3)

    while (delta > threshold):

        img_warped = affine_transform(It1, p_warped)

        b = It1 - img_warped

        b = b.reshape(b.size, 1)

        err = np.dot(A.T, b)

        result = np.linalg.solve(H, err)


        p1_delta = result[0, 0]

        p2_delta = result[1, 0]

        p3_delta = result[2, 0]

        p4_delta = result[3, 0]

        p5_delta = result[4, 0]

        p6_delta = result[5, 0]


        p_array = np.array([[1 + p1_delta, p3_delta, p5_delta], [p2_delta, 1 + p4_delta, p6_delta], [0, 0, 1]])

        p_inv = np.linalg.inv(p_array)

        p_warp = np.dot(p_warped, p_inv)


        flag = p_warp - I_matrix

        fin = np.array([[flag[0, 0]], [flag[1, 0]], [flag[0, 1]], [flag[1, 1]], [flag[0, 2]], [flag[1, 2]]])

        delta = np.square(np.linalg.norm(result))

        print (fin)

    return fin

#
# frames = np.load('../data/aerialseq.npy')
# img0= frames[:,:,0]
# img1= frames[:,:,1]
# M = InverseCompositionAffine(img0,img1)
# print (M)