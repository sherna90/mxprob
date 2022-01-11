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

import matplotlib.pyplot as plt

import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import pretrained_model, pretrained_models

from sklearn.metrics import classification_report
import h5py 

transform_train = transforms.Compose([
    transforms.RandomFlipLeftRight(),
    transforms.Resize([224,224]),
    transforms.ToTensor()
])

transform_test = transforms.Compose([
    transforms.Resize([224,224]),
    transforms.ToTensor()
])

training_path='~/code/PlantVillage-Dataset/data_distribution_for_SVM/train'
testing_path='~/code/PlantVillage-Dataset/data_distribution_for_SVM/test'

num_gpus = 1
model_ctx = mx.gpu()

num_workers = 10
batch_size = 32 

train_data = mx.gluon.data.vision.datasets.ImageFolderDataset(training_path).transform_first(transform_train)
test_data = mx.gluon.data.vision.datasets.ImageFolderDataset(testing_path).transform_first(transform_test)

train_data_loader = mx.gluon.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers)
val_data_loader = mx.gluon.data.DataLoader(test_data, batch_size, num_workers=num_workers)



hyper={'alpha':10.}
in_units=(3,224,224)
out_units=38

model=pretrained_model('resnet50_v2',hyper,in_units,out_units,ctx=model_ctx)
inference=sgd(model,step_size=0.001,ctx=model_ctx)

train_sgd=True
num_epochs=100

if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,chain_mame='resnet_map_plants.h5',data_loader=train_data_loader)
else:
    posterior_samples=h5py.File('resnet_map_plants.h5','r')
    par={var:posterior_samples[var][:] for var in posterior_samples.keys()}
    loss=posterior_samples.attrs['loss'][:]
    posterior_samples.close()

total_samples,total_labels,log_like=inference.predict(posterior_samples,data_loader=val_data_loader)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))