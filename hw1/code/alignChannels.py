import numpy as np
import matplotlib.pyplot as plt



def loss_calc(img1, img2):

    ## Sum of squared differences
    error= np.sum(np.square(img1-img2))
    i_init=0
    j_init=0

    for i in range(-30,30):

        new_img= np.roll(img2,i,axis=0)
        err_up= np.sum(np.square(img1-new_img))
        if err_up<error:
            error=err_up
            i_init=i

    for j in range(-30, 30):
        new_img = np.roll(new_img, j, axis=1)

        err_up = np.sum(np.square(img1 - new_img))  ## Sum of squared differences
        if err_up < error:
            error = err_up
            j_init = j


    return i_init, j_init



def alignChannels(red, green, blue):

    im_width= blue.shape[0]
    im_height= blue.shape[1]

    rgb_img= np.zeros((im_width,im_height,3),'uint8')


    ## Keeping blue channel and comparing blue channel with red and green to find the minimal loss

    rgb_img[:,:,2]=blue

    i,j= loss_calc(blue,green)
    print (i,j)

    green_img= np.roll(green, i,axis=0)
    green_img= np.roll(green_img,j, axis=1)

    u,v =loss_calc(blue,red)
    print (u,v)

    red_img= np.roll(red,u, axis=0)
    red_img= np.roll(red_img,v,axis=1)

    rgb_img[:, :, 1] = green_img
    rgb_img[:,:,0]= red_img


    return rgb_img

