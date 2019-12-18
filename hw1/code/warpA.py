
import numpy as np

def warp(im, A, output_shape):


    """ Warps (h,w) image im using affine (3,3) matrix A
    producing (output_shape[0], output_shape[1]) output image
    with warped = A*input, where warped spans 1...output_size.
    Uses nearest neighbor interpolation."""

    print (im.shape)
    print (output_shape)

    ## Since both the input greyscale image  and desired warped output image are of same dimension:

    out_img = np.zeros((im.shape[0],im.shape[1]))

    A_inv = np.linalg.inv(A)


    for h in range(out_img.shape[0]):
        for w in range(out_img.shape[1]):
            p_d = np.transpose(np.array([h, w, 1]))
            p_s = np.dot(A_inv, p_d)

            ## Float type should be converted to int to test boundary condtions

            h1 = int(p_s[0])
            w1 = int (p_s[1])

            ## Applying boundary conditions (rectangular range):

            if h1 in range(out_img.shape[0] - 1) and w1 in range(out_img.shape[1]-1):
                out_img[h][w] = im[h1][w1]

    return out_img
