#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../") 


import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from sklearn.metrics import classification_report
import h5py 

import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 360
import seaborn as sns

import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import lenet
from hamiltonian.inference.sgld import sgld
from hamiltonian.models.softmax import hierarchical_lenet
from hamiltonian.inference.sgld import hierarchical_sgld
from hamiltonian.utils.psis import *
from hamiltonian.inference.bbb import bbb


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.,1.)
])

num_gpus = 0
model_ctx = mx.cpu()
num_workers = 0
batch_size = 256 
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=True).transform_first(transform),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=False).transform_first(transform),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)


hyper={'alpha':10.}
in_units=(1,28,28)
out_units=10


print('#######################################')
print('Stochastic Gradient Descent')
model=lenet(hyper,in_units,out_units,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.1,ctx=model_ctx)

train_sgd=False
num_epochs=100
if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,
                           data_loader=train_data,chain_name='lenet_map.h5',verbose=True)
else:
    map_estimate=h5py.File('lenet_map.h5','r')
    par={var:nd.array(map_estimate[var][:]) for var in map_estimate.keys()}
    map_estimate.close()

total_samples,total_labels,log_like=inference.predict(par,batch_size=batch_size,num_samples=100,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

print('#######################################')
print('Bayes By Backprop')
model=lenet(hyper,in_units,out_units,ctx=model_ctx)
inference=bbb(model,par,step_size=0.005,ctx=model_ctx)

train_bbb=False
num_epochs=100

if train_bbb:
    par,loss,(means,sigmas)=inference.fit(epochs=num_epochs,batch_size=batch_size,
            data_loader=train_data,verbose=True,chain_name='lenet_variational.h5')
else:
    par=h5py.File('lenet_variational.h5','r')
    means={var:nd.array(par['means'][var][:],ctx=model_ctx) for var in par['means'].keys()}
    sigmas={var:nd.array(par['stds'][var][:],ctx=model_ctx) for var in par['stds'].keys()}

total_samples,total_labels,log_like=inference.predict(means,sigmas,batch_size=batch_size,num_samples=100,data_loader=val_data)
#y_hat=np.quantile(total_samples,.5,axis=0)
#print(classification_report(np.int32(total_labels),np.int32(y_hat)))


