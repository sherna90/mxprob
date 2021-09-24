
#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("../") 

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import resnet_softmax,hierarchical_resnet
from hamiltonian.inference.sgld import sgld
from hamiltonian.utils.psis import *

import matplotlib.pyplot as plt
import seaborn as sns
import glob
import h5py
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

transform_train = transforms.Compose([
    # Randomly crop an area and resize it to be 32x32, then pad it to be 40x40
    transforms.RandomCrop(32, pad=4),
    # Randomly flip the image horizontally
    transforms.RandomFlipLeftRight(),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation calculated across all images
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

model_ctx=mx.gpu(1)
per_device_batch_size = 256
num_workers = 16
batch_size = per_device_batch_size 

train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

# Set train=False for validation data
val_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)


hyper={'alpha':10.}
in_units=(3,32,32)
out_units=10
n_layers=18
pre_trained=False
num_epochs=250

print('#####################################################################################')
print('SGD Cifar')
model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.0001,ctx=model_ctx)

train_sgd=False
if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,
				data_loader=train_data,
        			chain_name='resnet_cifar_map.h5',
				verbose=True)

    fig=plt.figure(figsize=[5,5])
    plt.plot(loss,color='blue',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGD Resnet-18 CIFAR', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgd_cifar.pdf', bbox_inches='tight')
else:
    map_estimate=h5py.File('resnet_cifar_map.h5','r')
    par={var:map_estimate[var][:] for var in map_estimate.keys()}
    map_estimate.close()

total_samples,total_labels,log_like=inference.predict(par,batch_size=batch_size,
    num_samples=100,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))


# # Stochastic Gradient Langevin Dynamics Lenet <a class="anchor" id="chapter2"></a>
print('#####################################################################################')
print('SGLD Non-Hierarchical CIFAR')
model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.0001,ctx=model_ctx)

train_sgld=False

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,
				batch_size=batch_size,
                                data_loader=train_data,
                                verbose=True,
				chain_name='resnet_cifar_nonhierarchical.h5')
    plt.rcParams['figure.dpi'] = 360
    sns.set_style("whitegrid")
    fig=plt.figure(figsize=[5,5])
    plt.plot(loss[0],color='blue',lw=3)
    plt.plot(loss[1],color='red',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGLD Resnet-18 CIFAR', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgld_cifar.pdf', bbox_inches='tight')


posterior_samples=h5py.File('resnet_cifar_nonhierarchical.h5','r')
total_samples,total_labels,log_like=inference.predict(posterior_samples,
	data_loader=val_data)
posterior_samples.close()
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

# # Stochastic Gradient Langevin Dynamics Lenet <a class="anchor" id="chapter2"></a>
print('#####################################################################################')
print('SGLD Hierarchical CIFAR')
model=hierarchical_resnet(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.0001,ctx=model_ctx)

train_sgld=True

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,
				batch_size=batch_size,
                                data_loader=train_data,
                                verbose=True,
				chain_name='resnet_cifar_hierarchical.h5')
    plt.rcParams['figure.dpi'] = 360
    sns.set_style("whitegrid")
    fig=plt.figure(figsize=[5,5])
    plt.plot(loss[0],color='blue',lw=3)
    plt.plot(loss[1],color='red',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGLD Hierarchical Resnet-18 CIFAR', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgld_cifar.pdf', bbox_inches='tight')


posterior_samples=h5py.File('resnet_cifar_hierarchical.h5','r')
total_samples,total_labels,log_like=inference.predict(posterior_samples,
	data_loader=val_data)
posterior_samples.close()
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))
