#!/usr/bin/env python
# coding: utf-8

import sys

from numpy.core.fromnumeric import _mean_dispatcher
sys.path.append("../") 


import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import h5py 

import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import softmax,lenet,resnet_softmax
from hamiltonian.inference.sgld import sgld
from hamiltonian.inference.sgld import hierarchical_sgld
from hamiltonian.inference.bbb import bbb
from hamiltonian.utils.psis import *
import mxnet.gluon.probability as mxp
from mxnet import nd, autograd


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.,1.)
])

num_gpus = 0
model_ctx = mx.gpu()
num_workers = 2
batch_size = 32 
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=True).transform_first(transform),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=False).transform_first(transform),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)


hyper={'alpha':10.}
in_units=(1,28,28)
out_units=10

model=resnet_softmax(hyper,in_units,out_units,n_layers=18,ctx=model_ctx)

par=model.par
inference=hierarchical_sgld(model,model.par,step_size=1e-1,ctx=model_ctx)
scale_prior=mxp.HalfNormal(scale=1.)
stds={var:scale_prior.sample(par[var].shape).copyto(model_ctx) for var in par.keys()}
loc_prior=mxp.Normal(loc=0,scale=1.)
means={var:loc_prior.sample(par[var].shape).copyto(model_ctx) for var in par.keys()}
inference_sgd=sgd(model,par,step_size=0.1,ctx=model_ctx)


for var in par.keys():
    par[var].attach_grad()
    means[var].attach_grad()
    stds[var].attach_grad()

def SGD(params, lr):
    for var,param in params.items():
        param[:] = param - lr * param.grad

def SGLD(params, lr):
    loc_prior=mxp.Normal(loc=0.,scale=np.sqrt(lr))
    momentum={var:loc_prior.sample(params[var].shape).copyto(model_ctx) for var in params.keys()}
    for var,param in params.items():
        param[:] = param - lr * param.grad+momentum[var].as_nd_ndarray()

learning_rate=1e-3
iter=0
n_data=len(train_data)
for _ in range(10):
    for X,y in train_data:
        X=X.as_in_context(model_ctx)
        y=y.as_in_context(model_ctx)
        with autograd.record():     
            sigmas=dict()
            #par=dict()
            epsilons={var:loc_prior.sample(means[var].shape).copyto(model_ctx) for var in means.keys()}
            for var in means.keys():
                sigmas.update({var:inference.softplus(stds[var].as_nd_ndarray())})
                #par.update({var:means[var].as_nd_ndarray() + (sigmas[var].as_nd_ndarray() * epsilons[var].as_nd_ndarray())})
            loss = inference.centered_hierarchical_loss(par,means,epsilons,sigmas,X_train=X,y_train=y)
        loss.backward()
        SGLD(par, learning_rate)
        SGLD(means, learning_rate)
        SGLD(stds, learning_rate)
        curr_loss = nd.mean(loss).asscalar()
        #print('iter {0}, loss : {1}'.format(iter,curr_loss))
        iter+=1
        if (iter%100 == 0):
            for var in means.keys():
                means.update({var:means[var].as_nd_ndarray()})
            #total_samples,total_labels,log_like=inference.predict(means,sigmas,batch_size=batch_size,num_samples=10,data_loader=val_data)
            total_samples,total_labels,log_like=inference_sgd.predict(par,batch_size=batch_size,num_samples=100,data_loader=val_data)
            y_hat=np.quantile(total_samples,.5,axis=0)
            acc=accuracy_score(np.int32(total_labels),np.int32(y_hat)) 
            print('iter {0}, accuracy : {1:0.2f}, loss {2:0.2f}'.format(iter,acc,curr_loss))