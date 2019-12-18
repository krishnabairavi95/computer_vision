import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
from skimage.measure import compare_psnr as psnr
import numpy

train_data = scipy.io.loadmat('../data/data1/nist36_train_set1.mat')
valid_data = scipy.io.loadmat('../data/data1/nist36_valid_set1.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x = valid_data['valid_data']

dim = 32

# do PCA

u,s,vt = np.linalg.svd(train_x)

P_matrix = vt.T

projection_matrix = (np.array(P_matrix[:,:dim]))
print (projection_matrix.shape)

# rebuild a low-rank version
lrank = None

projection_x=  np.dot(train_x,projection_matrix)

rank= numpy.linalg.matrix_rank(projection_matrix)
print (rank)
# rebuild it
recon =  np.dot(projection_x,np.linalg.pinv(projection_matrix))


# for i in ([0,1,100,102,200,202,300,302,400,402]):
#     plt.subplot(2,1,1)
#     plt.imshow(train_x[i].reshape(32,32).T)
#     plt.subplot(2,1,2)
#     plt.imshow(recon[i].reshape(32,32).T)
#     plt.show()

# build valid dataset
recon_valid = None

projection_valid =  np.dot(valid_x,projection_matrix)
recon_valid=  np.dot(projection_valid,np.linalg.pinv(projection_matrix))


for i in ([0,1,100,102,200,202,300,302,400,402]):
    #print (i)
    plt.subplot(2,1,1)
    plt.imshow(valid_x[i].reshape(32,32).T)
    plt.subplot(2,1,2)
    plt.imshow(recon_valid[i].reshape(32,32).T)
    plt.show()
total = []

for pred,gt in zip(recon_valid,valid_x):
    total.append(psnr(gt,pred))
print(np.array(total).mean())