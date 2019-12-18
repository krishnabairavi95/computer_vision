import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from collections import Counter

train_data = scipy.io.loadmat('../data/data1/nist36_train_set1.mat')
valid_data = scipy.io.loadmat('../data/data1/nist36_valid_set1.mat')
#
# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']
print (valid_x.shape)

max_iters = 100
# pick a batch size, learning rate
batch_size = 36
learning_rate =  3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

# # initialize layers here
#


initialize_weights(1024,32,params,name='layer1')
m1= np.zeros((1024,32))
params['momentum_weight1'] = m1


initialize_weights(32,32,params,name='layer2')
m2= np.zeros((32,32))
params['momentum_weight2'] = m2



initialize_weights(32,32,params,name='layer3')
m3= np.zeros((32,32))
params['momentum_weight3'] = m3



initialize_weights(32,1024,params,name='output')
m4= np.zeros((32,1024))
params['momentum_weight4'] = m4


bias= np.zeros((32,))
bias_out= np.zeros((1024,))

params['momentum_bias1'] = bias
params['momentum_bias2'] = bias
params['momentum_bias3'] = bias
params['momentum_bias4'] = bias_out


fig=plt.figure()
# fig.show()
losslist = []
yforgraph = []


# should look like your previous training loops
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:

        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2


        ## Forward prop

        postact_layer1 = forward(xb,params,name='layer1',activation=relu)
        postact_layer2= forward(postact_layer1, params, name='layer2', activation=relu)
        postact_layer3 = forward(postact_layer2, params, name='layer3', activation=relu)
        forward_out=  forward(postact_layer3, params, name='output', activation=sigmoid)

        tse_loss = np.square(xb- forward_out)
        loss_itr = np.sum(tse_loss)
        total_loss += loss_itr

        ## Backward prop

        delta = 2* (forward_out- xb)

        delta_1 =  backwards(delta,params,  name='output',activation_deriv=sigmoid_deriv)
        delta_2 = backwards(delta_1, params, name='layer3', activation_deriv= relu_deriv)
        delta_3 = backwards(delta_2, params, name='layer2', activation_deriv=relu_deriv)

        delta_4 = backwards(delta_3, params, name='layer1', activation_deriv=relu_deriv)

      # valid_loss = np.sum (tse_loss)/ valid_x. shape[0]

        ## Implementing momentum weight initialization:

        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions

        for k,v in sorted(list(params.items())):
            if 'grad' in k:
                name = k.split('_')[1]
                params['momentum_' + name] = 0.9*params['momentum_' + name] - learning_rate*params[k]
                params[name] = params[name] + params['momentum_' + name]

    total_loss = total_loss / (batch_size * batch_num)
#
    losslist.append(total_loss)
    yforgraph.append(itr)
    plt.plot(yforgraph, losslist, 'b')
    plt.xlabel(" Epochs")
    plt.ylabel (" Loss")
    #plt.scatter(yforgraph,losslist)

    postact_layer1 = forward(valid_x, params, name='layer1', activation=relu)
    postact_layer2 = forward(postact_layer1, params, name='layer2', activation=relu)
    postact_layer3 = forward(postact_layer2, params, name='layer3', activation=relu)
    forward_out = forward(postact_layer3, params, name='output', activation=sigmoid)

    tse = np.square (valid_x - forward_out)

    loss_valid = np.sum(tse) / 3600
#
#
    if itr % 2 == 0:
        print("epoch: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9

plt.show()
#
# visualize some results
# Q5.3.1

xb =  valid_x

import matplotlib.pyplot as plt
h1 = forward(xb,params,'layer1',relu)
h2 = forward(h1,params,'layer2',relu)
h3 = forward(h2,params,'layer3',relu)
out = forward(h3,params,'output',sigmoid)

for i in ([0,1,100,102,200,202,300,302,400,402]):
    #print (i)
    plt.subplot(2,1,1)
    plt.imshow(xb[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(out[i].reshape(32,32).T)
    plt.show()


from skimage.measure import compare_psnr as psnr

# evaluate PSNR
# Q5.3.2


psnrlist = []
length = xb.shape[0]
for i in range(length):
    originalImage = xb[i].reshape(32, 32).T
    obtainedImage = out[i].reshape(32, 32).T
    value = psnr(originalImage, obtainedImage)
    psnrlist.append(value)

psnrlist = np.array(psnrlist)
avgpsnr = np.mean(psnrlist)
print("average psnr is {:.2f}".format(avgpsnr))

