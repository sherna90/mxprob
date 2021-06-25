#!/usr/bin/env python


import sys
sys.path.append("../") 


import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms




transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.,1.)
])



num_gpus = 1
model_ctx = mx.gpu()

num_workers = 8
batch_size = 64 
train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=True).transform_first(transform),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=False).transform_first(transform),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)




import mxnet as mx
from mxnet import nd, autograd, gluon

hyper={'alpha':10.}
in_units=(28,28)
out_units=10


from hamiltonian.inference.sgld import sgld
from hamiltonian.models.softmax import lenet

model=lenet(hyper,in_units,out_units,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.1,ctx=model_ctx)

par,loss,posterior_samples=inference.fit_gluon(epochs=200,batch_size=batch_size,
                             data_loader=train_data,
                             verbose=True)


total_samples,total_labels=inference.predict(posterior_samples,5,data_loader=val_data)


from sklearn.metrics import classification_report
posterior_samples

y_hat=np.quantile(total_samples,.9,axis=0)

print(classification_report(np.int32(total_labels),np.int32(y_hat)))


