import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

#
train_data = scipy.io.loadmat('../data/data1/nist36_train_set1.mat')
valid_data = scipy.io.loadmat('../data/data1/nist36_valid_set1.mat')
test_data = scipy.io.loadmat('../data/data1/nist36_test_set1.mat')


# train_data = scipy.io.loadmat('../data/data2/nist36_train_set2.mat')
# valid_data = scipy.io.loadmat('../data/data2/nist36_valid_set2.mat')
# test_data = scipy.io.loadmat('../data/data2/nist36_test_set2.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

#print (test_x.shape)

# y_count=[]
#
# for i in train_y:
#     i = np.argmax(i)
#     y_count.append (i)
#
# print (y_count)




avg_acc = 0
max_iters = 50
# pick a batch size, learning rate
batch_size = 30
learning_rate = 5e-3
hidden_size = 64
classes= 36

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(train_x.shape[1],hidden_size,params,'layer1')
initialize_weights(hidden_size,classes,params,'output')

train_loss=[]
val_loss=[]
train_acc=[]
val_acc=[]


init_weights = params['Wlayer1']

#

#fig1 = plt.figure()

#grid = ImageGrid(fig1, 111, nrows_ncols=(8,8,),axes_pad=0.0)

# for i in range(init_weights.shape[1]):
#     grid[i].imshow(init_weights[:,i].reshape((32,32)))
# plt.show()



# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # forward
        hlayer = forward(xb, params, 'layer1')
        probs = forward(hlayer, params, 'output', softmax)
        # loss
        # be sure to add loss and accuracy to epoch totals
        loss, acc = compute_loss_and_acc(yb, probs)
        avg_acc += acc
        total_loss += loss
        # backward

        err = probs - yb
        deriv = backwards(err, params, 'output', linear_deriv)
        backwards(deriv, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Wlayer1'] = params['Wlayer1'] - (learning_rate * params['grad_Wlayer1'])
        params['blayer1'] = params['blayer1'] - (learning_rate * params['grad_blayer1'])
        params['Woutput'] = params['Woutput'] - (learning_rate * params['grad_Woutput'])
        params['boutput'] = params['boutput'] - (learning_rate * params['grad_boutput'])

    batch_length = len(batches)

    avg_acc = avg_acc / batch_length

    avg_loss= total_loss/ batch_length


    # training loop can be exactly the same as q2!

    # if itr % 2 == 0:
    #     print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))

    # run on validation set and report accuracy! should be above 75%
    val_hidden= forward(valid_x, params, 'layer1')
    val_probs = forward(val_hidden, params, 'output', softmax)
    valid_loss, valid_acc = compute_loss_and_acc(valid_y, val_probs)



    val_loss. append (valid_loss/ valid_x.shape[0])
    val_acc.append(valid_acc)

    train_loss.append(total_loss/train_x.shape[0])
    train_acc.append (avg_acc)

    print("itr: {:02d} \t Training Accuracy {:.2f} \t Validation accuracy: {:.2f}".format( itr,avg_acc,valid_acc))
    # print (" Training Accuracy",avg_acc)
    # print('Validation accuracy: ', valid_acc)
    # #
    if False: # view the data
        for crop in xb:
            import matplotlib.pyplot as plt
            plt.imshow(crop.reshape(32,32).T)
            plt.show()


import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

#
# plt.figure(0)
# plt.plot(np.arange(max_iters), train_acc, 'r')
# plt.plot(np.arange(max_iters), val_acc, 'b')
# plt.legend(['training accuracy', 'validation accuracy'])
# plt.ylabel('Accuracy')
# plt.xlabel('Epochs')
# plt.show()
# #plt.figure(1)
#
# plt.figure(0)
# plt.plot(np.arange(max_iters), train_loss, 'r')
# plt.plot(np.arange(max_iters), val_loss, 'b')
# plt.legend(['training loss', 'validation loss'])
# plt.ylabel('Loss')
# plt.xlabel('Epochs')
# plt.show()


test_hidden= forward(test_x, params, 'layer1')
test_probs = forward(test_hidden, params, 'output', softmax)
test_loss, test_acc = compute_loss_and_acc(test_y, test_probs)

print ("============= Test accuracy=================")

print(test_acc)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

init_weights = params['Wlayer1']

fig1 = plt.figure()

grid = ImageGrid(fig1, 111, nrows_ncols=(8,8,),axes_pad=0.0)

# for i in range(init_weights.shape[1]):
#     grid[i].imshow(init_weights[:,i].reshape((32,32)))
# plt.show()


# Q3.1.3
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
val_hidden = forward(valid_x, params, 'layer1')
val_probs = forward(val_hidden, params, 'output', softmax)

for c in range(valid_y.shape[0]):
    pred_true= np.argmax(val_probs[c,:])
    out_true=  np.argmax(valid_y[c,:])
    confusion_matrix[pred_true, out_true] +=1


# import string
# plt.imshow(confusion_matrix,interpolation='nearest')
# plt.grid(True)
# plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.show()





