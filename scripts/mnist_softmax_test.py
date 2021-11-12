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
from hamiltonian.models.softmax import softmax,lenet
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
inference=sgd(model,model.par,step_size=0.001,ctx=model_ctx)

par=model.par
inference=bbb(model,model.par,step_size=1e-1,ctx=model_ctx)
scale_prior=mxp.HalfNormal(scale=1.)
stds={var:scale_prior.sample(par[var].shape).copyto(model_ctx) for var in par.keys()}
loc_prior=mxp.Normal(loc=0,scale=1.)
means={var:loc_prior.sample(par[var].shape).copyto(model_ctx) for var in par.keys()}
inference_sgd=sgd(model,par,step_size=0.1,ctx=model_ctx)
epsilons={var:loc_prior.sample(par[var].shape).copyto(model_ctx) for var in par.keys()}

for var in par.keys():
    means[var].attach_grad()
    stds[var].attach_grad()
    epsilons[var].attach_grad()

def SGD(params,batch_size, lr):
    for var,param in params.items():
        param[:] = param - lr * param.grad/(1.*batch_size)

def softplus(x):
    return mx.np.log1p(x)

learning_rate=1e-2
iter=0
n_data=len(train_data)
for _ in range(100):
    for X,y in train_data:
        X=X.as_in_context(model_ctx)
        y=y.as_in_context(model_ctx)
        with autograd.record():     
            sigmas=dict()
            par=dict()
            for var in means.keys():
                sigmas.update({var:softplus(stds[var])})
                par.update({var:means[var] + (sigmas[var] * epsilons[var])})                
            loss = inference.variational_loss(par,means,epsilons,sigmas,n_data,batch_size,X_train=X,y_train=y)
        loss.backward()
        SGD(epsilons,batch_size, learning_rate)
        SGD(means,batch_size, learning_rate)
        SGD(stds,batch_size, learning_rate)
        curr_loss = nd.mean(loss).asscalar()
        #print('iter {0}, loss : {1}'.format(iter,curr_loss))
        iter+=1
        if (iter%100 == 0):
            #for var in means.keys():
            #    means.update({var:par[var]})
            #total_samples,total_labels,log_like=inference.predict(means,sigmas,batch_size=batch_size,num_samples=10,data_loader=val_data)
            total_samples,total_labels,log_like=inference_sgd.predict(par,batch_size=batch_size,num_samples=100,data_loader=val_data)
            y_hat=np.quantile(total_samples,.5,axis=0)
            acc=accuracy_score(np.int32(total_labels),np.int32(y_hat)) 
            print('iter {0}, accuracy : {1:0.2f}, loss {2:0.2f}'.format(iter,acc,curr_loss))