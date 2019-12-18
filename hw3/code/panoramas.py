import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from planarH import ransacH
from BRIEF import briefLite,briefMatch,plotMatches

def imageStitching(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix

    INPUT
        Warps img2 into img1 reference frame using the provided warpH() function
        H2to1 - a 3 x 3 matrix encoding the homography that best matches the linear
                 equation
    OUTPUT
        Blends img1 and warped img2 and outputs the panorama image
    '''
    #######################################
    # TO DO ...

    ## Warping im2 into im1 reference

    im2_warped=cv2.warpPerspective(im2,H2to1,(im1.shape[1]*2,im1.shape[0]))
   # cv2.imwrite('../results/6.1.jpg', im2_warped)
    #cv2.imshow('WarpedImage2',im2_warped)


   ## Blending it

    new_im1 =np.zeros((im2.shape[0],im1.shape[1]*2,3))
    new_im1[:, 0:im1.shape[1], :] = im1
    new_im1 =np.uint8(new_im1)
    pano_im=np.maximum(im2_warped,new_im1)

    cv2.imshow('pan0_im', pano_im)
    cv2.waitKey(0)
    cv2.imwrite('../results/q6_1.jpg', pano_im)
    return pano_im




def imageStitching_noClip(im1, im2, H2to1):
    '''
    Returns a panorama of im1 and im2 using the given 
    homography matrix without cliping.
    ''' 
    ######################################
    # TO DO ...

    M_matrix= np.array([[1.0,0.0,0.0],[0.0,1.,130],[0.0,0.0,1.0]])

    corner1=  [im2.shape[1]-1,0,1]
    corner2= [0,im2.shape[0]-1,1]
    corner3=  [im2.shape[1],im2.shape[0],1]
    corner4=  [0,0,1]

    corner1= np.inner(H2to1, np.array(corner1))
    corner2 = np.inner(H2to1, np.array(corner2))
    corner3 = np.inner(H2to1, np.array(corner3))
    corner4 = np.inner(H2to1, np.array(corner4))


    c1 = np.rint(corner1/corner1[-1])
    c2= np.rint(corner2/corner2[-1])
    c3= np.rint(corner3/corner3[-1])
    c4= np.rint(corner4/corner4[-1])


    max_height= np.amax((im1.shape[1],c1[1],c2[1],c3[1],c4[1]))
    min_height = np.amin((0, c1[1], c2[1], c3[1], c4[1]))

    height= int(max_height-min_height)

    max_width= np.amax((im1.shape[0],c1[0],c2[0],c3[0],c4[0]))

    min_width= np.amin((0,c1[0],c2[0],c3[0],c4[0]))

    width= int(max_width-min_width)

    stitch1=cv2.warpPerspective(im1, M_matrix, (width,height))

    stitch2=cv2.warpPerspective(im2, np.matmul(M_matrix,H2to1), (width,height))

    pano_im = np.maximum(stitch1,stitch2)

    cv2.imshow('pan0_im', pano_im)
    cv2.waitKey(0)
    cv2.imwrite('../results/q6_2_pan.jpg', pano_im)
    return pano_im


def generatePanorama(im1, im2):
    '''
    Returns a panorama of im1 and im2 without cliping.
    ''' 
    ######################################
    # TO DO ...

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    pano_im = imageStitching_noClip(im1, im2, H2to1)
    cv2.imwrite('../results/q6_3.jpg', pano_im)
    cv2.imshow('panoramas', pano_im)
    return pano_im


if __name__ == '__main__':
    im1 = cv2.imread('../data/incline_L.png')
    im2 = cv2.imread('../data/incline_R.png')

    locs1, desc1 = briefLite(im1)
    locs2, desc2 = briefLite(im2)
    matches = briefMatch(desc1, desc2)
    H2to1 = ransacH(matches, locs1, locs2, num_iter=5000, tol=2)
    np.save('../results/q6_1.npy', H2to1)
    imageStitching_noClip(im1, im2, H2to1)
    pano_im= generatePanorama(im1, im2)



