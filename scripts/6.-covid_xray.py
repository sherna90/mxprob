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
import h5py 

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score


transform = transforms.Compose([
    transforms.Resize([150,150]),
    transforms.ToTensor(),
    transforms.Normalize(mean=0,std=1)
])


training_path='/media/sergio/Backup/data/COVID-19_Radiography_Dataset/split_data/train'
testing_path='/media/sergio/Backup/data/COVID-19_Radiography_Dataset/split_data/val'


num_gpus = 1
model_ctx = mx.gpu()
num_workers = 4
batch_size = 64 
train_data = mx.gluon.data.vision.datasets.ImageFolderDataset(training_path).transform_first(transform)
test_data = mx.gluon.data.vision.datasets.ImageFolderDataset(testing_path).transform_first(transform)
train_data_loader = mx.gluon.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers)
valid_data_loader = mx.gluon.data.DataLoader(test_data, batch_size, num_workers=num_workers)


# # Bayesian inference for Covid X-Ray
# 
# * [Stochastic Gradient Descent](#chapter1)
# * [Stochastic Gradient Langevin Dynamics](#chapter2)
# 
# 

# # Stochastic Gradient Descent <a class="anchor" id="chapter1"></a>



hyper={'alpha':10.}
in_units=(3,150,150)
out_units=4
n_layers=18
pre_trained=True


print('#####################################################################################')
print('SGD Resnet')

model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.001,ctx=model_ctx)
train_sgd=True
num_epochs=250

if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,
    data_loader=train_data_loader,chain_name='resnet_map.h5',verbose=True)

    fig=plt.figure(figsize=[5,5])
    plt.plot(loss,color='blue',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGD Resnet Covid', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgd_covid.pdf', bbox_inches='tight')
else:
    map_estimate=h5py.File('resnet_map.h5','r')
    par={var:map_estimate[var][:] for var in map_estimate.keys()}
    map_estimate.close()

total_samples,total_labels,log_like=inference.predict(par,batch_size=batch_size,
    num_samples=100,data_loader=valid_data_loader)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))
score=[]
for q in np.arange(.1,.9,.1):
    y_hat=np.quantile(total_samples,q,axis=0)
    score.append(f1_score(np.int32(total_labels),np.int32(y_hat), average='macro'))
print('mean f-1 : {0}, std f-1 : {1}'.format(np.mean(score),2*np.std(score)))


# # Stochastic Gradient Langevin Dynamics <a class="anchor" id="chapter2"></a>

print('#####################################################################################')
print('SGLD Resnet')

from hamiltonian.inference.sgld import sgld
from hamiltonian.models.softmax import resnet_softmax

model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.01,ctx=model_ctx)

train_sgld=True
num_epochs=250

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                                data_loader=train_data_loader,
                                verbose=True,chain_name='resnet_posterior_nonhierarchical.h5')

    plt.rcParams['figure.dpi'] = 360
    sns.set_style("whitegrid")
    fig=plt.figure(figsize=[5,5])
    plt.plot(loss[0],color='blue',lw=3)
    plt.plot(loss[1],color='red',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGLD Resnet COVID', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgld_covid.pdf', bbox_inches='tight')


posterior_samples=h5py.File('resnet_posterior_nonhierarchical.h5','r')
total_samples,total_labels,log_like=inference.predict(posterior_samples,data_loader=valid_data_loader)
posterior_samples.close()
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))



print('#####################################################################################')
print('SGLD Hierarchical Resnet')

from hamiltonian.inference.sgld import sgld
from hamiltonian.models.softmax import hierarchical_resnet

model=hierarchical_resnet(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.01,ctx=model_ctx)

train_sgld=True
num_epochs=250

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                                data_loader=train_data_loader,
                                verbose=True,chain_name='resnet_posterior_hierarchical.h5')

    plt.rcParams['figure.dpi'] = 360
    sns.set_style("whitegrid")
    fig=plt.figure(figsize=[5,5])
    plt.plot(loss[0],color='blue',lw=3)
    plt.plot(loss[1],color='red',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGLD Resnet COVID', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgld_covid.pdf', bbox_inches='tight')


posterior_samples=h5py.File('resnet_posterior_hierarchical.h5','r')
total_samples,total_labels,log_like=inference.predict(posterior_samples,data_loader=valid_data_loader)
posterior_samples.close()
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))
