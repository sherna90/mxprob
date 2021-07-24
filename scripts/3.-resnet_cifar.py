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
from hamiltonian.models.softmax import resnet_softmax
from hamiltonian.inference.sgld import sgld
from hamiltonian.utils.psis import *

import matplotlib.pyplot as plt
import seaborn as sns
import glob

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

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

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

model_ctx=mx.cpu()
per_device_batch_size = 512
num_workers = 0
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
pre_trained=True
train_sgd=True
num_epochs=3

print('#####################################################################################')
print('SGD Cifar')
model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.0001,ctx=model_ctx)
if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,data_loader=train_data,verbose=True)

    fig=plt.figure(figsize=[5,5])
    plt.plot(loss,color='blue',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGD Resnet-18 CIFAR', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgd_cifar.pdf', bbox_inches='tight')
    model.net.save_parameters('results/cifar/resnet_sgd_'+str(num_epochs)+'_epochs.params')
else:
    model.net.load_parameters('results/cifar/resnet_sgd_'+str(num_epochs)+'_epochs.params',ctx=model_ctx)
    par=dict()
    for name,gluon_par in model.net.collect_params().items():
        par.update({name:gluon_par.data()})

total_samples,total_labels,log_like=inference.predict(par,batch_size=batch_size,num_samples=100,data_loader=val_data)
score=[]
for q in np.arange(.1,.9,.1):
    y_hat=np.quantile(total_samples,q,axis=0)
    score.append(f1_score(np.int32(total_labels),np.int32(y_hat), average='macro'))
print("SGD")
print('mean f-1 : {0}, std f-1 : {1}'.format(np.mean(score),2*np.std(score)))


# # Stochastic Gradient Langevin Dynamics Lenet <a class="anchor" id="chapter2"></a>
print('#####################################################################################')
print('SGLD Non-Hierarchical CIFAR')
model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.0001,ctx=model_ctx)

train_sgld=True
num_epochs=3

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
    plt.title('SGLD Resnet-18 CIFAR', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgld_cifar.pdf', bbox_inches='tight')
else:
    chain1=glob.glob("results/cifar/chain_nonhierarchical_0_1_sgld*")
    chain2=glob.glob("results/cifar/chain_nonhierarchical_0_sgld*")
    posterior_samples=[chain1,chain2]


posterior_samples_flat=[item for sublist in posterior_samples for item in sublist]
total_samples,total_labels,log_like=inference.predict(posterior_samples_flat,5,data_loader=val_data)

score=[]
for q in np.arange(.1,.9,.1):
    y_hat=np.quantile(total_samples,q,axis=0)
    score.append(f1_score(np.int32(total_labels),np.int32(y_hat), average='macro'))
print('mean f-1 : {0}, std f-1 : {1}'.format(np.mean(score),2*np.std(score)))

