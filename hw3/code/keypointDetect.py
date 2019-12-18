import numpy as np
import cv2
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.morphology import generate_binary_structure


def createGaussianPyramid(im, sigma0=1, 
        k=np.sqrt(2), levels=[-1,0,1,2,3,4]):

    if len(im.shape)==3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.max()>10:
        im = np.float32(im)/255
    im_pyramid = []
    for i in levels:
        sigma_ = sigma0*k**i 
        im_pyramid.append(cv2.GaussianBlur(im, (0,0), sigma_))
    im_pyramid = np.stack(im_pyramid, axis=-1)
    return im_pyramid

def displayPyramid(im_pyramid):
    im_pyramid = np.split(im_pyramid, im_pyramid.shape[2], axis=2)
    im_pyramid = np.concatenate(im_pyramid, axis=1)
    im_pyramid = cv2.normalize(im_pyramid, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    cv2.imshow('Pyramid of image', im_pyramid)
    cv2.waitKey(0) # press any key to exit
    cv2.destroyAllWindows()

def createDoGPyramid(gaussian_pyramid, levels=[-1,0,1,2,3,4]):
    '''
    Produces DoG Pyramid
    Inputs
    Gaussian Pyramid - A matrix of grayscale images of size
                        [imH, imW, len(levels)]
    levels      - the levels of the pyramid where the blur at each level is
                   outputs
    DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
                   created by differencing the Gaussian Pyramid input
    '''
    DoG_pyramid = []
    ################
    # TO DO ...
    # compute DoG_pyramid here
    DoG_levels = levels[1:]

    for i in range(1,len(DoG_levels)+1):
        DoG_pyramid.append(gaussian_pyramid[:,:,i]-gaussian_pyramid[:,:,i-1])
    DoG_pyramid = np.stack(DoG_pyramid, axis=-1)
    return DoG_pyramid, DoG_levels

def computePrincipalCurvature(DoG_pyramid):
    '''
    Takes in DoGPyramid generated in createDoGPyramid and returns
    PrincipalCurvature,a matrix of the same size where each point contains the
    curvature ratio R for the corre-sponding point in the DoG pyramid
    
    INPUTS
        DoG Pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
    
    OUTPUTS
        principal_curvature - size (imH, imW, len(levels) - 1) matrix where each 
                          point contains the curvature ratio R for the 
                          corresponding point in the DoG pyramid
    '''

    principal_curvature = np.zeros((DoG_pyramid.shape[0],DoG_pyramid.shape[1],DoG_pyramid.shape[2]))

    # TO DO ...
    # Compute principal curvature here


    levels= DoG_pyramid.shape[2]
    for i in range(levels):
        dx=cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,1,0)
        dxx= cv2.Sobel(dx,cv2.CV_64F,1,0)
        dxy= cv2.Sobel(dx,cv2.CV_64F,0,1)
        dy= cv2.Sobel(DoG_pyramid[:,:,i],cv2.CV_64F,0,1)
        dyy = cv2.Sobel(dy, cv2.CV_64F, 0, 1)
        #dyx= cv2.Sobel(dy, cv2.CV_64F, 1, 0)

        # From Hessian Matrix
        trace_h= dxx+dyy
        det_h= dxx*dyy- (dxy**2)

        R= np.square(trace_h)/det_h
        principal_curvature[:,:,i]=R

    return principal_curvature

def getLocalExtrema(DoG_pyramid, DoG_levels, principal_curvature,
        th_contrast=0.03, th_r=12):
    '''
    Returns local extrema points in both scale and space using the DoGPyramid

    INPUTS
        DoG_pyramid - size (imH, imW, len(levels) - 1) matrix of the DoG pyramid
        DoG_levels  - The levels of the pyramid where the blur at each level is
                      outputs
        principal_curvature - size (imH, imW, len(levels) - 1) matrix contains the
                      curvature ratio R
        th_contrast - remove any point that is a local extremum but does not have a
                      DoG response magnitude above this threshold
        th_r        - remove any edge-like points that have too large a principal
                      curvature ratio
     OUTPUTS
        locsDoG - N x 3 matrix where the DoG pyramid achieves a local extrema in both
               scale and space, and also satisfies the two thresholds.
    '''
    nhood = generate_binary_structure(2, 2)
    locsDoG=[]

    for i in range(len(DoG_levels)):
        DoG_check= abs(DoG_pyramid[:,:,i]>th_contrast)
        curvature= abs(principal_curvature[:,:,i]<th_r)

        if i == 0 or i==(len(DoG_levels)-1):
            minbef = np.ones(DoG_pyramid[:, :, i].shape, dtype=bool)
            maxbef= minbef
            minafter= minbef
            maxafter= minbef

        else:
            minbef = DoG_pyramid[:, :, i] < DoG_pyramid[:, :, i - 1]
            maxbef= DoG_pyramid[:, :, i] > DoG_pyramid[:, :, i - 1]
            minafter = DoG_pyramid[:, :, i] < DoG_pyramid[:, :, i + 1]
            maxafter = DoG_pyramid[:,  :, i] > DoG_pyramid[:, :, i + 1]


        layer1 = maximum_filter(DoG_pyramid[:,:,i], footprint=nhood, mode='constant')  == DoG_pyramid[:, :, i]

        layer2= minimum_filter(DoG_pyramid[:,:,i], footprint=nhood, mode='constant') == DoG_pyramid[:, :, i]


        main_points=  DoG_check*curvature*(layer1*maxbef*maxafter)+ DoG_check*curvature*(layer2*minbef*minafter)

        loc1, loc2 = np.where(main_points == True)
        print (loc1.shape)

        level_loc = i * np.ones(np.shape(loc1))
        locs_all = np.hstack(((loc1.reshape( (-1, 1)), loc2.reshape((-1, 1)), level_loc.reshape((-1, 1)))))
        locsDoG.append(locs_all)

    locsDoG = (np.vstack(locsDoG)).astype(int)

    return locsDoG


def DoGdetector(im, sigma0=1, k=np.sqrt(2), levels=[-1,0,1,2,3,4],
                th_contrast=0.03, th_r=12):
    '''
    Putting it all together

    Inputs          Description
    --------------------------------------------------------------------------
    im              Grayscale image with range [0,1].

    sigma0          Scale of the 0th image pyramid.

    k               Pyramid Factor.  Suggest sqrt(2).

    levels          Levels of pyramid to construct. Suggest -1:4.

    th_contrast     DoG contrast threshold.  Suggest 0.03.

    th_r            Principal Ratio threshold.  Suggest 12.

    Outputs         Description
    --------------------------------------------------------------------------

    locsDoG         N x 3 matrix where the DoG pyramid achieves a local extrema
                    in both scale and space, and satisfies the two thresholds.

    gauss_pyramid   A matrix of grayscale images of size (imH,imW,len(levels))
    '''
    ##########################
    # TO DO ....
    # compupte gauss_pyramid, gauss_pyramid here

    gauss_pyramid=createGaussianPyramid(im)
    DoG_pyr, DoG_levels = createDoGPyramid(gauss_pyramid, levels)
    pc_curvature = computePrincipalCurvature(DoG_pyr)

    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)

    return locsDoG, gauss_pyramid




if __name__ == '__main__':
    # test gaussian pyramid
    levels = [-1,0,1,2,3,4]
    im = cv2.imread('../data/model_chickenbroth.jpg')


    im_pyr = createGaussianPyramid(im)
   # displayPyramid(im_pyr)

   ## test DoG pyramid

    DoG_pyr, DoG_levels = createDoGPyramid(im_pyr, levels)
    #displayPyramid(DoG_pyr)
   #
   #  # test compute principal curvature


    pc_curvature = computePrincipalCurvature(DoG_pyr)
    th_contrast = 0.03
    th_r = 12
    locsDoG = getLocalExtrema(DoG_pyr, DoG_levels, pc_curvature, th_contrast, th_r)


    # test DoG detector


    locsDoG, gaussian_pyramid = DoGdetector(im)

    # print (locsDoG.shape)
    #
    # for t in range(locsDoG.shape[0]):
    #     print (locsDoG[t,3])
    #     break


    #print (locsDoG[1, 0].shape)

    #print (locsDoG[1, 0])

    for i in locsDoG:
        cv2.circle(im, (i[1], i[0]), 1, (0, 255, 0), -1)
    cv2.imshow('Image.jpg', im)
    cv2.imwrite('../results/keypoints.png', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()