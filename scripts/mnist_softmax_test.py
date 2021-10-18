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

import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import softmax
from hamiltonian.inference.sgld import sgld
from hamiltonian.models.softmax import hierarchical_softmax
from hamiltonian.inference.sgld import hierarchical_sgld
from hamiltonian.utils.psis import *
import mxnet.gluon.probability as mxp
from mxnet import nd, autograd


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
in_units=(28,28)
out_units=10


map_estimate=h5py.File('mnist_map.h5','r')
par={var:map_estimate[var][:] for var in map_estimate.keys()}
map_estimate.close()

hierarchical_model=hierarchical_softmax(hyper,in_units,out_units,ctx=model_ctx)
inference=hierarchical_sgld(hierarchical_model,par,step_size=0.001,ctx=model_ctx)

prior=mxp.HalfNormal(scale=1.0)
stds={var:prior.sample(par[var].shape).copyto(model_ctx) for var in par.keys()}
for var in par.keys():
    par[var]=nd.array(par[var])
    par[var].attach_grad()
    stds[var].attach_grad()

for X,y in train_data:
    break

with autograd.record():     
    loss = inference.loss(par,stds,X_train=X,y_train=y)
loss.backward()