"""

Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import scipy.ndimage as ndimage
import helper
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# Insert your package here


'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''



def help_func(pts1,pts2):
    u1 = pts1[:, 0]
    u2 = pts2[:, 0]

    v1 = pts1[:, 1]
    v2 = pts2[:, 1]

    W_1 = np.multiply(u1, u2)
    W_2 = np.multiply(v1, u2)
    W_3 = u2
    W_4 = np.multiply(u1, v2)
    W_5 = np.multiply(v1, v2)
    W_6 = v2
    W_7 = u1
    W_8 = v1
    W_9 = np.ones((1, pts1.shape[0]))

    W = np.vstack((W_1, W_2, W_3, W_4, W_5, W_6, W_7, W_8, W_9))
    W = np.transpose(W)
    return W

def eightpoint(pts1, pts2, M):
    # Replace pass by your implementation

    # Normalizing

    pts1= pts1 /M
    pts2= pts2 /M

    W=  help_func(pts1,pts2)
    u, s, vh = np.linalg.svd(W)

    eigen_value = vh[-1, :]
    eigen_value=eigen_value.reshape(3, 3)

    F = helper._singularize(eigen_value)

    F = helper.refineF(F, pts1, pts2)

    M_list=[[1/M,0,0],[0,1/M,0],[0,0,1]]

    T_matrix= np.asarray(M_list)

    F= np.matmul(T_matrix.T,np.matmul(F,T_matrix))

    np.savez('q2_1.npz', F=F, M=M)

    return F



'''
Q2.2: Seven Point Algorithm
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: Farray, a list of estimated fundamental matrix.
'''
def sevenpoint(pts1, pts2, M):
    # Replace pass by your implementation
    pts1= pts1 /M
    pts2= pts2 /M

    W = help_func(pts1, pts2)
    u, s, vh = np.linalg.svd(W)

    e1=  vh[-1, :]
    e2 = vh[-2, :]

    e1 = e1.reshape(3, 3)
    e2 =  e2.reshape(3, 3)

    F1=  helper.refineF(e1, pts1, pts2)
    F2=  helper.refineF(e2, pts1, pts2)

    def det(alpha):
        out= np.linalg.det(alpha * F1 + (1 - alpha) * F2)
        return out

    res1= det(0)
    res2= 2*(det(1)-det(-1))/3-(det(2)-det(-2))/12
    res3= 0.5*(det(1))+0.5*(det(-1)-det(0))
    res4= det(1)-res1-res2-res3

    res=[res4,res3,res2,res1]

    M_list=[[1/M,0,0],[0,1/M,0],[0,0,1]]

    T_matrix= np.asarray(M_list)

    roots= np.roots(res)


    Fstack=[]
    for alpha in roots:
        if np.isreal(alpha)==True:
            new_F= alpha * F1 + (1-alpha)*F2
            F=  np.matmul(T_matrix.T,np.matmul(new_F,T_matrix))
            Fstack.append(F)

    return Fstack



'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # Replace pass by your implementation
    E = np.matmul(K2.T, np.matmul(F, K1))
    return E


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    # Replace pass by your implementation

    W=[]

    for j in range(len(pts1)):
        A1 = pts1[j, 0] * C1[2, :] - C1[0, :]
        A2 = pts1[j, 1] * C1[2, :] - C1[1, :]
        A3 = pts2[j, 0] * C2[2, :] - C2[0, :]
        A4 = pts2[j, 1] * C2[2, :] - C2[1, :]
        A = np.vstack((A1, A2, A3, A4))

        u,s,vh=np.linalg.svd(A)

        p= vh[-1,:]
        p =p/p[3]
        W.append(p)

    W= np.asarray(W)
    w= W[:,:3]

    new_p1= np.matmul(C1,np.transpose(W))

    new_p2 = np.matmul(C2, np.transpose(W))

    new_p1=new_p1/new_p1[-1,:]
    new_p2=new_p2/new_p2[-1,:]

    err1 = np.linalg.norm(new_p1[0:2, :].T - pts1) ** 2
    err2 = np.linalg.norm(new_p2[0:2, :].T - pts2) ** 2

    err= err1 +err2

    return w,err

'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1):
    # Replace pass by your implementation

    arr= np.array([x1,y1,1])
    dot_prod = np.dot(F,arr)

    window=8
    _range= round(max(im1.shape[0],im1.shape[1])/16)


    y_range= np.arange(y1-_range, y1+_range)
    x_range =  -((dot_prod[1] * y_range) + dot_prod[2]) / dot_prod[0]

    patch1 = im1[y1 - window:y1 + window + 1, x1 - window:x1 + window + 1, :]
    patch1 = ndimage.gaussian_filter(patch1, sigma=4)
    min_err = 1000**2
    cor = 0

    H=im1.shape[0]
    W=im1.shape[1]
    for i in range(_range*2):

        y2 = y_range[i]
        x2 = int(x_range[i])

        if (x2 >= window and x2 <=W - window - 1 and y2 >= window and y2 <= H - window - 1):
            patch2 = im2[y2 - window:y2 + window + 1, x2 - window:x2 + window + 1, :]
            patch2 = ndimage.gaussian_filter(patch2, sigma=4)
            patch= patch1-patch2
            err = np.linalg.norm(patch)
            if err < min_err:
                min_err = err
                corr = i

    return x_range[corr], y_range[corr]



# if __name__ == "__main__":
#
#
#     some_corresp = np.load('../data/some_corresp.npz')
#     pts1= some_corresp['pts1']
#     pts2 = some_corresp['pts2']
#
#     im1 = plt.imread('../data/im1.png')
#     im2 = plt.imread('../data/im2.png')
#
#     intrinsics = np.load('../data/intrinsics.npz')
#     K1 = intrinsics['K1']
#     K2 = intrinsics['K2']
#
#
#     h, w, _ = np.shape(im1)
#     M = max(h, w)
#
#
#     # Visualizing eight points:
#
#     F = eightpoint(pts1, pts2, M)
#     E= essentialMatrix(F, K1, K2)
#     #
#     # temple_coords = np.load('../data/templeCoords.npz')
#     # x1 = temple_coords['x1']
#     # y1 = temple_coords['y1']
#
#    # np.savez('q4_1.npz',F=F, pts1=pts1, pts2=pts2)
#     #helper.epipolarMatchGUI(im1,im2,F)
#     # print (" ========== The Essential Matrix is ==========")
#     # print (E)
#     # # print(" ===== F matrix for eight-point algorithm is ============= ")
#     # print (F_array)
#     helper.displayEpipolarF(im1, im2, F)
#
#
#     ## Visualizing seven points
#     #
#     # F_array = sevenpoint(pts1, pts2, M)
#     # #
#     # for i in F_array:
#     #     helper.displayEpipolarF(im1, im2, i)
#     # #
#     # best_F= F_array[1]
#     # print (" ===== Best F_array for seven-point algorithm is ============= ")
#     # print (best_F)
#     # np.savez('q2_2.npz', F = F_array[1], M = M, pts1 = pts1, pts2 = pts2)
#     # # helper.displayEpipolarF(im1, im2,best_F)
#     #
