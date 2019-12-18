import numpy as np
import threading
import queue
import imageio
import os,time
import math
import visual_words
import skimage
import multiprocessing



def features_all_imgs(data, dictionary, layers, K):

    params=[]
    for i in range(len(data)):
        param= [data[i], dictionary, layers, K]
        params.append(param)
    return params


def build_recognition_system(num_workers=4):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * num_workers: number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''


    train_data = np.load("../data/train_data.npz")
    dictionary = np.load("dictionary.npy")
    # ----- TODO -----

    layers = 2

    print (train_data.files)
    print ("done")
    data = train_data['files']
    labels = train_data['labels']
    K= dictionary.shape[0]

    pool = multiprocessing.Pool(processes=num_workers)

    params= features_all_imgs(data, dictionary, layers, K)

    features = pool.map(get_image_feature, params)

    print ("reached")
    np.savez("trained_system.npz", features=features, dictionary=dictionary, labels=labels, SPM_layer_num= layers)
    print ("succeeded")



def evaluate_recognition_system(num_workers=4):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * num_workers: number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''


    test_data = np.load("../data/test_data.npz")
    trained_system = np.load("trained_system.npz")
    # ----- TODO -----
    #print(test_data.files)
    test_imgs = test_data['files']

    test_labels = test_data['labels']

    train_labels= trained_system["labels"]
    dictionary= trained_system["dictionary"]

    print (np.shape(dictionary))

    train_features= trained_system["features"]

    K = dictionary.shape[0]

    layers = 2

    num_classes= 8

    confusion_matrix = np.zeros((num_classes, num_classes))

    pool = multiprocessing.Pool(processes=num_workers)
    params = features_all_imgs(test_imgs, dictionary, layers, K)

    test_features = pool.map(get_image_feature, params)


    for i in range(len(test_imgs)):

        dist = distance_to_set(test_features[i], train_features)

        confusion_matrix[test_labels[i], train_labels[np.argmax(dist)]] = confusion_matrix[test_labels[i], train_labels[np.argmax(dist)]] +1

        accuracy = np.sum(np.trace(confusion_matrix)) / len(test_imgs)

    print ("===================Results===================")
    print (confusion_matrix)
    print ("====================Accuracy=====================")
    print (accuracy)


    return  confusion_matrix, accuracy



def get_image_feature(args):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * file_path: path of image file to read
    * dictionary: numpy.ndarray of shape (K,3F)
    * layer_num: number of spatial pyramid layers
    * K: number of clusters for the word maps

    [output]
    * feature: numpy.ndarray of shape (K*(4^(L+1)/3))
    '''

    # ----- TODO -----

    [file_path, dictionary, layer_num, K] = args

    img = "../data/" + file_path

    img= skimage.io.imread(img)

    print (type(img))


    wordmap= visual_words.get_visual_words(img,dictionary)

    feature_SPM= get_feature_from_wordmap_SPM(wordmap,layer_num,K)

    return feature_SPM

def distance_to_set(word_hist,histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''
    # ----- TODO -----

    min_dist = np.minimum(word_hist, histograms)
    return np.sum(min_dist, axis=1)


def get_feature_from_wordmap(wordmap,dict_size):
    '''
    Compute histogram of visual words.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * dict_size: dictionary size K

    [output]
    * hist: numpy.ndarray of shape (K)
    '''
    
    # ----- TODO -----
    hist= np.histogram(wordmap,bins=dict_size+1)

    return hist[0]

def get_feature_from_wordmap_SPM(wordmap,layer_num,dict_size):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * wordmap: numpy.ndarray of shape (H,W)
    * layer_num: number of spatial pyramid layers
    * dict_size: dictionary size K

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^layer_num-1)/3) (as given in the write-up)
    '''
    
    # ----- TODO -----



    hist_all=[]

    weight=[0.25,0.25,0.5]


    for l in range(layer_num+1):

        rowwise= np.array_split(wordmap,2**l,axis=0)

        for row in rowwise:
            colwise=np.array_split(row,2**l,axis=1)

            for col in colwise:
                hist= get_feature_from_wordmap(col,dict_size)
                hist_norm= hist/wordmap.shape[0]*wordmap.shape[1]* weight[l]
                hist_all = np.append(hist_all, hist_norm)

    return hist_all


if __name__ == "__main__":

    #dictionary = np.load("dictionary.npy")
    #wordmap.imread('/Users/krishnabairavi/Desktop/Fall19/CV/hw2/code/waterfall_1.jpg')


   # hist_all=get_feature_from_wordmap_SPM(wordmap,2,dictionary.size)

   # build_recognition_system(num_workers=2)

    # confusion_matrix,accuracy= evaluate_recognition_system(num_workers=4)
    #
    # print ("===================Results===================")
    # print (confusion_matrix)
    # print ("====================Accuracy=====================")
    # print (accuracy)
