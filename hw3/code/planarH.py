import numpy as np
import cv2
from BRIEF import briefLite, briefMatch
import random

def computeH(p1, p2):
    '''
    INPUTS:
        p1 and p2 - Each are size (2 x N) matrices of corresponding (x, y)'  
                 coordinates between two images
    OUTPUTS:
     H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear 
            equation
    '''
    assert(p1.shape[1]==p2.shape[1])
    assert(p1.shape[0]==2)
    #############################
    # TO DO ...

    X1= p1[0, :]
    Y1= p1[1, :]
    U1=p2[0, :]
    V1= p2[1, :]


    X1 = X1.reshape((X1.size, 1))
    Y1 = Y1.reshape((Y1.size, 1))
    U1 = U1.reshape((U1.size, 1))
    V1 = V1.reshape((V1.size, 1))

    N = p1.shape[1]

    vector_one = np.concatenate((-U1,-V1,-1*np.ones((N,1)),np.zeros((N,3)), X1*U1,X1*V1,X1), axis = 1)
    vector_two= np.concatenate((np.zeros((N,3)),-U1,-V1,-1*np.ones((N,1)),np.multiply(Y1,U1),Y1*V1,Y1), axis = 1)

    A_matrix= np.vstack((vector_one,vector_two))

    ## Computing H from A using SVD

    u, c, vector = np.linalg.svd(A_matrix)

    H = vector[-1:]

    print(H.shape)
    H2to1 = H.reshape((3, 3))

    return H2to1

def ransacH(matches, locs1, locs2, num_iter=5000, tol=2):
    '''
    Returns the best homography by computing the best set of matches using
    RANSAC
    INPUTS
        locs1 and locs2 - matrices specifying point locations in each of the images
        matches - matrix specifying matches between these two sets of point locations
        nIter - number of iterations to run RANSAC
        tol - tolerance value for considering a point to be an inlier

    OUTPUTS
        bestH - homography matrix with the most inliers found during RANSAC
    ''' 
    ###########################
    # TO DO ...

    degree_of_freedom= 8
    pair_points= int(degree_of_freedom/2)

    ## Finding desired matches based on the locs and matches


    match_loc1=locs1[matches[:,0],0:2]
    match_loc2= locs2[matches[:,1],0:2]

    match_loc1= np.transpose(match_loc1[:, 0:2])
    match_loc2=np.transpose(match_loc2 [:, 0:2])


    num_points= match_loc1.shape[1]
    print (num_points)

    dest1= np.vstack((match_loc1,np.ones((1,num_points))))
    dest2=np.vstack((match_loc2,np.ones((1,num_points))))

    max_inlier=-1
    bestH=[]

    for iter in range(num_iter):

        idx = np.zeros(pair_points)

        for j in range(pair_points):

            idx[j] = random.randint(0, num_points - 1)

        idx = idx.astype(int)

        p1 = match_loc1[:,idx]
        p2 = match_loc2[:,idx]

        H = computeH(p1, p2)

        new_dest=np.matmul(H,dest2)

        ## Dividing by the scaling factor (last row) to make the last row 1

        new_dest=new_dest/new_dest[-1,:]

        c_dist= new_dest-dest1

        dist= np.linalg.norm (c_dist,axis=0)
        inlier=np.sum(dist<2)

        if (inlier > max_inlier):
            max_inlier=inlier
            bestH=H

    return bestH


if __name__ == '__main__':
    im1 = cv2.imread('../data/model_chickenbroth.jpg')
    im2 = cv2.imread('../data/chickenbroth_01.jpg')
    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)

    bestH = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)

