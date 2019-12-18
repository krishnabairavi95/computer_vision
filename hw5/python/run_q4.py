# # # #
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.patches as mpatches
import torch
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings

os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

import pickle
import string

letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
params = pickle.load(open('q3_weights.pickle', 'rb'))


def identify( bw, line, output, prev_end):

    output_pred = ""

    line.sort(key=lambda x: x[1])

    for letter in line:


        img = bw[letter[0]:letter[2], letter[1]:letter[3]]

        if (letter[1] - prev_end) > 0.75 * (letter[3] - letter[1]):
            output += ' '

        img = np.pad(img, ((30, 30), (30, 30)), 'constant', constant_values=0.0)


        from skimage import img_as_float

        img = skimage.transform.resize(img_as_float(img), (32, 32), anti_aliasing=True)

        # #
        img = skimage.morphology.dilation(img_as_float(img), skimage.morphology.square(3))

        img = 1 - img

        input_image = img_as_float(img).T

        input = np.reshape(img_as_float(input_image), (1, 32 * 32))


        h1 = forward(input, params, 'layer1')

        probs = forward(h1, params, 'output', softmax)

        pred = np.argmax(probs)

        output_pred += letters[pred]

    output_pred += '\n'

    print('Identified text  : \n{}'.format(output_pred))


#
for img in os.listdir('../images'):

    ## Avoding the inbuilt .DS Store file

    if not img.startswith('.'):

        print('Image: {}'.format(img))

        im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))

        bboxes, bw = findLetters(im1)

        bbox_util= bboxes[0]

        line=[]

        line.append(bbox_util[:])

        overall_lines=[]

        for i in range(len(bboxes)-1):

            if bboxes[i+1][0]<bboxes[i][2]:
                line.append(bboxes[i+1][:])


            elif bboxes[i+1][0]>bboxes[i][2]:
                #
                overall_lines.append(line)

                line.sort(key=lambda x: x[1])

                prev_end = line[0][3]

                output = " "
                for letter in line:

                    img = bw[letter[0]:letter[2], letter[1]:letter[3]]

                    if (letter[1] - prev_end) > 0.75 * (letter[3] - letter[1]):
                        output += ' '

                    img = np.pad(img, ((30, 30), (30, 30)), 'constant', constant_values=0.0)

                    # plt.imshow(img)
                    # plt.show()
                    # #

                    prev_end = letter[3]

                    #
                    from skimage import img_as_float

                    img = skimage.transform.resize(img_as_float(img), (32, 32), anti_aliasing=True)
                    # plt.imshow(img)
                    # plt.show()

                    # #
                    img = skimage.morphology.dilation(img_as_float(img), skimage.morphology.square(3))
                    # plt.imshow(img)
                    # plt.show()

                    img = 1 - img
                    input_image = img_as_float(img).T

                    input = np.reshape(img_as_float(input_image), (1, 32 * 32))
                    # plt.imshow(img)
                    # plt.show()

                    ### Loading the feature engineered, resized image into NN for prediction

                    h1 = forward(input, params, 'layer1')
                    probs = forward(h1, params, 'output', softmax)

                    pred = np.argmax(probs)

                    output += letters[pred]

                output += '\n'

                print('Recognised text : \n{}'.format(output))
                line = []
                line.append(bboxes[i + 1][:])

                line = []
                line.append(bboxes[i+1][:])

            else:

                overall_lines.append(line)


    line.sort(key=lambda x: x[1])

    prev_end = line[0][3]

    output = " "
    for letter in line:

        img = bw[letter[0]:letter[2], letter[1]:letter[3]]

        if (letter[1] - prev_end) > 0.75 * (letter[3] - letter[1]):
            output += ' '

        img = np.pad(img, ((30, 30), (30, 30)), 'constant', constant_values=0.0)

        # plt.imshow(img)
        # plt.show()
        # #

        prev_end = letter[3]

        #
        from skimage import img_as_float

        img = skimage.transform.resize(img_as_float(img), (32, 32), anti_aliasing=True)
        # plt.imshow(img)
        # plt.show()

        # #
        img = skimage.morphology.dilation(img_as_float(img), skimage.morphology.square(3))
        # plt.imshow(img)
        # plt.show()

        img = 1 - img
        input_image = img_as_float(img).T

        input = np.reshape(img_as_float(input_image), (1, 32 * 32))
        # plt.imshow(img)
        # plt.show()

        ### Loading the feature engineered, resized image into NN for prediction

        h1 = forward(input, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)

        pred = np.argmax(probs)

        output += letters[pred]

    output += '\n'

    print('Recognised text : \n{}'.format(output))
    line = []
    line.append(bboxes[i + 1][:])



    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                            fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)

    plt.show()

