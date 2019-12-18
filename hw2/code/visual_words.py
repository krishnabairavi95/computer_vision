import numpy as np
import multiprocessing
import imageio
import scipy.ndimage
import skimage.color
import sklearn.cluster
import scipy.spatial.distance
import os,time
import util
import random
import math
import matplotlib.pyplot as plt




def extract_filter_responses(image):
    '''
    Extracts the filter responses for the given image.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''

    # ----- TODO -----

    if np.amax(image)>1:
        image= np.float32(image)/255

    if image.ndim<3:
        image= np.tile(image[:,np.expand_dims],(1,1,3))

    if image.ndim>3:
        image= image[:,:,0:3]


    image=  skimage.color.rgb2lab(image)

    channels= 3

    for s in [1,2,4,8, np.sqrt(2)*8]:

        for ch in range(channels):

            new_img = scipy.ndimage.gaussian_filter(image[:, :, ch], s)

            if s==1 and ch==0:
                imgs= new_img[:,:, np.newaxis]

            else:
                imgs= np.dstack((imgs,new_img[:,:,np.newaxis]))

            new_img = scipy.ndimage.gaussian_laplace(image[:, :, ch], s)
            imgs = np.dstack((imgs, new_img[:, :, np.newaxis]))

            new_img = scipy.ndimage.gaussian_filter(image[:, :, ch], s, order=[0, 1])
            imgs = np.dstack((imgs, new_img[:, :, np.newaxis]))

            new_img = scipy.ndimage.gaussian_filter(image[:, :, ch], s, order=[1, 0])
            imgs = np.dstack((imgs, new_img[:, :, np.newaxis]))

    return imgs



def reshape_to_2D(image):
    filter_responses = extract_filter_responses(image)
    print(filter_responses.shape)
    T = filter_responses.shape[0] * filter_responses.shape[1]
    F = filter_responses.shape[2]
    structured_response = np.reshape(filter_responses, (T, F))

    return structured_response



def get_visual_words(image,dictionary):
    '''
    Compute visual words mapping for the given image using the dictionary of visual words.

    [input]
    * image: numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    
    # ----- TODO -----
    print (dictionary.shape)
    filter_responses= extract_filter_responses(image)
    structured_response= reshape_to_2D(image)
    dist = scipy.spatial.distance.cdist(structured_response, dictionary)
    wordmap = np.argmin(dist, axis=1)
    wordmap = wordmap.reshape(filter_responses.shape[0], filter_responses.shape[1])
    print(wordmap.shape)
    return wordmap

def compute_dictionary_one_image(args):
    '''
    Extracts random samples of the dictionary entries from an image.
    This is a function run by a subprocess.

    [input]
    * i: index of training image
    * alpha: number of random samples
    * image_path: path of image file
    * time_start: time stamp of start time

    [saved]
    * sampled_response: numpy.ndarray of shape (alpha,3F)
    '''


    i,alpha,image_path = args
    # ----- TODO -----

    image = skimage.io.imread(image_path)

    filter_responses =  extract_filter_responses(image)
    print (filter_responses.shape)
    print ("HIIIII ....................................")
    T= filter_responses.shape[0]*filter_responses.shape[1]

    ## Making 3D matrix to be 2D

    filter_responses=reshape_to_2D(image)

    random_pixels=np.random.sample(range(1, T), alpha)
    print (random_pixels)

    sampled_pixels=filter_responses[random_pixels,:]

    return sampled_pixels




def compute_dictionary(num_workers=4):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * num_workers: number of workers to process in parallel
    
    [saved]
    * dictionary: numpy.ndarray of shape (K,3F)
    NOTE : Please save the dictionary as 'dictionary.npy' in the same dir as the code.
    '''

    train_data = np.load("../data/train_data.npz")
    print (train_data.files)
    print ("done")
    # ----- TODO -----

    images = train_data['files']
    ground = train_data['labels']

    print (type(images))
    print (len(images))
    print ("Hii")

    alpha=200
    num_imgs= len(images)
    filters=60

    main_dict= np.zeros((num_imgs,alpha,filters))
    folder_path= "../data/"

    for img in range(num_imgs):
        path =  folder_path + images[img]
        args = (img, alpha, path)
        small_dict= compute_dictionary_one_image(args)
        main_dict[img,:,:]= small_dict

    ## Reshaping it to alpha*T, 3F sizer

    reshaped_dict = np.reshape(main_dict, (num_imgs * alpha, filters))
    kmeans = sklearn.cluster.KMeans(n_clusters=150).fit(reshaped_dict)
    dictionary = kmeans.cluster_centers_

    np.save('dictionary.npy',dictionary)



# if __name__ == "__main__":
#
#     input= '../data/waterfall/sun_abcxnrzizjgcwkdn.jpg'
#
#
#     image= skimage.io.imread(input)
#
#     dictionary= np.load('dictionary.npy')
#
#     out= get_visual_words(image, dictionary)
#     print (out)
#     plt.imshow(skimage.color.label2rgb(out))
#     #plt.imshow(out)
#     plt.show()
#
#     plt.imsave('waterfall_3_new.jpg',skimage.color.label2rgb(out))


   # out= extract_filter_responses(image)
    #print (out.shape)
    #util.display_filter_responses(out)

   #compute_dictionary(num_workers=2)

   #print (out.shape)

