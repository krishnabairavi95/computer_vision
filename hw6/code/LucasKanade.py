import numpy as np
from scipy.interpolate import RectBivariateSpline

def LucasKanade(It, It1, rect, p0 = np.zeros(2)):
	# Input:
	# 	It: template image
	# 	It1: Current image
	# 	rect: Current position of the car
	# 	(top left, bot right coordinates)
	# 	p0: Initial movement vector [dp_x0, dp_y0]
	# Output:
	# 	p: movement vector [dp_x, dp_y]
    #
    # Put your implementation here

    delta = 100

    threshold = 0.05

    p = p0


    x1,y1,x2,y2 = rect

    width = x2 - x1
    height = y2 - y1


    rows_it=  range (0, It.shape[0])
    cols_it = range(0, It.shape[1])

    rows_it1=  range (0, It1.shape[0])
    cols_it1 = range(0, It1.shape[1])


    spline_it = RectBivariateSpline(rows_it, cols_it, It)

    spline_it1 = RectBivariateSpline(rows_it1, cols_it1, It1)

    x, y = np.mgrid[x1: x2 + 1: width * 1j, y1: y2 + 1: height * 1j]

    reference = spline_it.ev(y, x).flatten()


    while (np.linalg.norm(delta) > threshold):


        img= spline_it1.ev(y+ p[0], x+ p[1]).flatten()

        error= reference-img


        ## Evaluation

        grad_x =  spline_it1.ev(y+ p[0], x + p[1], dy=1). flatten()

        grad_y = spline_it1.ev(y+ p[0], x + p[1], dx=1). flatten()

        deriv = np.vstack ((grad_x, grad_y))

        invs =  np.linalg.pinv (deriv)

        delta= np.matmul(invs.T, error)

        p += delta

    return p

