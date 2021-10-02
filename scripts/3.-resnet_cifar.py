#!/usr/bin/env python
# coding: utf-8

# # Uncertainty estimates from CIFAR 10

# In[1]:


import sys
sys.path.append("../") 


# In[3]:


from __future__ import division

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms


# In[4]:


transform_train = transforms.Compose([
    # Randomly crop an area and resize it to be 32x32, then pad it to be 40x40
    #transforms.RandomCrop(32, pad=4),
    # Randomly flip the image horizontally
    #transforms.RandomFlipLeftRight(),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation calculated across all images
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


# In[6]:


transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


# In[10]:


# Batch Size for Each GPU
model_ctx=mx.cpu()
per_device_batch_size = 512
# Number of data loader workers
num_workers = 0
# Calculate effective total batch size
batch_size = per_device_batch_size 

# Set train=True for training data
# Set shuffle=True to shuffle the training data
train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

# Set train=False for validation data
val_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)


# In[11]:


for X,y in train_data:
    print(X.shape)
    print(y.shape)
    break


# # SGD Resnet 18

# In[12]:


import mxnet as mx
from mxnet import nd, autograd, gluon

hyper={'alpha':10.}
in_units=(3,32,32)
out_units=10
n_layers=18
pre_trained=True


# In[15]:


import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import resnet_softmax

model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.0001,ctx=model_ctx)


# In[14]:


import hamiltonian
import importlib

try:
    importlib.reload(hamiltonian.models.softmax)
    importlib.reload(hamiltonian.inference.sgd)
    print('modules re-loaded')
except:
    print('no modules loaded yet')


# In[ ]:


par,loss=inference.fit(epochs=5,batch_size=batch_size,data_loader=train_data,verbose=True)


# In[16]:


import matplotlib.pyplot as plt

plt.plot(loss)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[17]:


total_samples,total_labels,log_like=inference.predict(par,batch_size=batch_size,num_samples=100,data_loader=val_data)   


# In[18]:


y_hat=np.quantile(total_samples,.5,axis=0)


# In[19]:


from sklearn.metrics import classification_report

print(classification_report(np.int32(total_labels),np.int32(y_hat)))


# # SGLD

# In[8]:


import mxnet as mx
from hamiltonian.inference.sgld import sgld
from hamiltonian.models.softmax import resnet_softmax

model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.0001,ctx=model_ctx)


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
import glob

train_sgld=True
num_epochs=20

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                                data_loader=train_data,
                                verbose=True,chain_name='chain_nonhierarchical')

    plt.rcParams['figure.dpi'] = 360
    sns.set_style("whitegrid")
    fig=plt.figure(figsize=[5,5])
    plt.plot(loss[0],color='blue',lw=3)
    plt.plot(loss[1],color='red',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGLD Cifar MNIST', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgld_cifar.pdf', bbox_inches='tight')
else:
    chain1=glob.glob("../scripts/results/cifar/chain_nonhierarchical_0_1_sgld*")
    chain2=glob.glob("../scripts/results/cifar/chain_nonhierarchical_0_sgld*")
    posterior_samples=[chain1,chain2]


# In[ ]:




