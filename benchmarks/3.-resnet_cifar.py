
import sys
sys.path.append("../") 

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms




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


#num_gpus = mx.context.num_gpus()
#model_ctx = [mx.gpu(i) for i in range(num_gpus)]

per_device_batch_size = 128
# Number of data loader workers
num_workers = 8
# Calculate effective total batch size
batch_size = per_device_batch_size * 1

# Set train=True for training data
# Set shuffle=True to shuffle the training data
train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

# Set train=False for validation data
val_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)




import mxnet as mx
from mxnet import nd, autograd, gluon

model_ctx=mx.gpu()
hyper={'alpha':10.}
in_units=(32,32)
out_units=10
n_layers=18
pre_trained=False



import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import resnet_softmax

model=resnet_softmax(hyper,in_units,out_units,n_layers,pre_trained,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.0001,ctx=model_ctx)



par,loss=inference.fit(epochs=250,batch_size=batch_size,data_loader=train_data,verbose=True)




total_labels=[]
total_samples=[]
for X_test,y_test in val_data:
    X_test=X_test.as_in_context(model_ctx)
    y_hat=model.predict(model.par,X_test)
    if X_test.shape[0]==batch_size:
        samples=[]
        for _ in range(100):
            samples.append(y_hat.sample().asnumpy())
        total_samples.append(np.asarray(samples))
        total_labels.append(y_test.asnumpy())
total_samples=np.concatenate(total_samples,axis=1)
total_labels=np.concatenate(total_labels)    




y_hat=np.quantile(total_samples,.1,axis=0)




from sklearn.metrics import classification_report

print(classification_report(np.int32(total_labels),np.int32(y_hat)))






