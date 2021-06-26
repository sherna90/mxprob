#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append("../") 

import argparse, time, logging, random, math

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


# In[4]:


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


import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import softmax

model=softmax(hyper,in_units,out_units,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.001,ctx=model_ctx)
par,loss=inference.fit(epochs=100,batch_size=batch_size,data_loader=train_data,verbose=True)


import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 360
sns.set_style("whitegrid")


fig=plt.figure(figsize=[5,5])
plt.plot(loss,color='blue',lw=3)
plt.xlabel('Epoch', size=18)
plt.ylabel('Loss', size=18)
plt.title('SGD Softmax MNIST', size=18)
plt.xticks(size=14)
plt.yticks(size=14)
plt.savefig('sgd_softmax.pdf', bbox_inches='tight')


model.net.save_parameters('softmax_sgd_100_epochs.params')




model.net.load_parameters('softmax_sgd_100_epochs.params',ctx=model_ctx)
par=dict()
for name,gluon_par in model.net.collect_params().items():
    par.update({name:gluon_par.data()})
               
total_samples,total_labels=inference.predict(par,batch_size=batch_size,num_samples=10,data_loader=val_data)


y_hat=np.quantile(total_samples,.5,axis=0)
from sklearn.metrics import classification_report
print(classification_report(np.int32(total_labels),np.int32(y_hat)))


# # Stochastic Gradient Langevin Dynamics <a class="anchor" id="chapter2"></a>



from hamiltonian.inference.sgld import sgld

model=softmax(hyper,in_units,out_units,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.01,ctx=model_ctx)
loss,posterior_samples=inference.sample(epochs=100,batch_size=batch_size,
                             data_loader=train_data,
                             verbose=True)

import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.dpi'] = 360
sns.set_style("whitegrid")


fig=plt.figure(figsize=[5,5])
plt.plot(loss[0],color='blue',lw=3)
plt.plot(loss[1],color='red',lw=3)
plt.xlabel('Epoch', size=18)
plt.ylabel('Loss', size=18)
plt.title('SGLD Softmax MNIST', size=18)
plt.xticks(size=14)
plt.yticks(size=14)
plt.savefig('sgld_softmax.pdf', bbox_inches='tight')


posterior_samples_flat=[item for sublist in posterior_samples for item in sublist]
total_samples,total_labels=inference.predict(posterior_samples_flat,5,data_loader=val_data)


from sklearn.metrics import classification_report
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))


