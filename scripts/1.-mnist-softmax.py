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
import seaborn as sns
plt.rcParams['figure.dpi'] = 360
sns.set_style("whitegrid")

import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.inference.sgld import sgld
from hamiltonian.models.softmax import softmax,hierarchical_softmax

from sklearn.metrics import classification_report
import arviz as az

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.,1.)
])

num_gpus = 1
model_ctx = mx.gpu()
num_epochs=100
num_workers = 2
batch_size = 64 
train_sgd=False

train_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=True).transform_first(transform),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

val_data = gluon.data.DataLoader(
    gluon.data.vision.MNIST(train=False).transform_first(transform),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)



hyper={'alpha':10.}
in_units=(28,28)
out_units=10



model=softmax(hyper,in_units,out_units,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.001,ctx=model_ctx)



if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,data_loader=train_data,verbose=True)

    fig=plt.figure(figsize=[5,5])
    plt.plot(loss,color='blue',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGD Softmax MNIST', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgd_softmax.pdf', bbox_inches='tight')
    model.net.save_parameters('softmax_sgd_'+str(num_epochs)+'_epochs.params')
else:
    model.net.load_parameters('softmax_sgd_'+str(num_epochs)+'_epochs.params',ctx=model_ctx)
    par=dict()
    for name,gluon_par in model.net.collect_params().items():
        par.update({name:gluon_par.data()})
               
total_samples,total_labels=inference.predict(par,batch_size=batch_size,num_samples=10,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))


# # Stochastic Gradient Langevin Dynamics Softmax <a class="anchor" id="chapter2"></a>
print('#####################################################################################')
model=softmax(hyper,in_units,out_units,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.01,ctx=model_ctx)
loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                             data_loader=train_data,
                             verbose=True,chain_name='chain_nonhierarchical')

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

y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

posterior_samples_multiple_chains=inference.posterior_diagnostics(posterior_samples)
datasets=[az.convert_to_inference_data(sample) for sample in posterior_samples_multiple_chains]
dataset = az.concat(datasets, dim="chain")
mean_r_hat_values={var:float(az.rhat(dataset)[var].mean().data) for var in model.par}
mean_ess_values={var:float(az.ess(dataset)[var].mean().data) for var in model.par}
mean_mcse_values={var:float(az.mcse(dataset)[var].mean().data) for var in model.par}

print('R_hat')
print("\n".join("{}\t{}".format(k, v) for k, v in mean_r_hat_values.items()))
print('ESS')
print("\n".join("{}\t{}".format(k, v) for k, v in mean_ess_values.items()))
print('MCSE')
print("\n".join("{}\t{}".format(k, v) for k, v in mean_mcse_values.items()))

# # Stochastic Gradient Langevin Dynamics Hierarchical <a class="anchor" id="chapter3"></a>
print('#####################################################################################')
model=hierarchical_softmax(hyper,in_units,out_units,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.01,ctx=model_ctx)
loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                             data_loader=train_data,
                             verbose=True,chain_name='chain_hierarchical')


fig=plt.figure(figsize=[5,5])
plt.plot(loss[0],color='blue',lw=3)
plt.plot(loss[1],color='red',lw=3)
plt.xlabel('Epoch', size=18)
plt.ylabel('Loss', size=18)
plt.title('SGLD Hierarchical Softmax MNIST', size=18)
plt.xticks(size=14)
plt.yticks(size=14)
plt.savefig('sgld_hierarchical_softmax.pdf', bbox_inches='tight')


posterior_samples_flat=[item for sublist in posterior_samples for item in sublist]
total_samples,total_labels=inference.predict(posterior_samples_flat,5,data_loader=val_data)

y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

posterior_samples_multiple_chains=inference.posterior_diagnostics(posterior_samples)
datasets=[az.convert_to_inference_data(sample) for sample in posterior_samples_multiple_chains]
dataset = az.concat(datasets, dim="chain")
mean_r_hat_values={var:float(az.rhat(dataset)[var].mean().data) for var in model.par}
mean_ess_values={var:float(az.ess(dataset)[var].mean().data) for var in model.par}
mean_mcse_values={var:float(az.mcse(dataset)[var].mean().data) for var in model.par}

print('R_hat')
print("\n".join("{}\t{}".format(k, v) for k, v in mean_r_hat_values.items()))
print('ESS')
print("\n".join("{}\t{}".format(k, v) for k, v in mean_ess_values.items()))
print('MCSE')
print("\n".join("{}\t{}".format(k, v) for k, v in mean_mcse_values.items()))
