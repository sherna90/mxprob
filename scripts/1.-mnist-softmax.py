#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append("../") 
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from sklearn.metrics import classification_report,f1_score
import h5py 

import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import softmax
from hamiltonian.inference.sgld import sgld
from hamiltonian.inference.sgld import hierarchical_sgld
from hamiltonian.utils.psis import *
from hamiltonian.utils.utils import *

import tensorflow as tf
import tensorflow_probability as tfp
ess_estimate = lambda samples : tfp.mcmc.diagnostic.effective_sample_size(samples, filter_beyond_positive_pairs=False,cross_chain_dims=1).numpy()
r_hat_estimate = lambda samples : tfp.mcmc.diagnostic.potential_scale_reduction(samples, independent_chain_ndims=1,split_chains=False).numpy()


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
model=softmax(hyper,in_units,out_units,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.001,ctx=model_ctx)

train_sgd=False
num_epochs=100
if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,
                           data_loader=train_data,chain_name='mnist_map.h5',verbose=True)

else:
    map_estimate=h5py.File('mnist_map.h5','r')
    par={var:mx.np.array(map_estimate[var][:],ctx=model_ctx) for var in map_estimate.keys()}
    map_estimate.close()

total_samples,total_labels,log_like=inference.predict(par,batch_size=batch_size,num_samples=100,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

print('#######################################')
print('Stochastic Gradient Langevin Dynamics')
inference=sgld(model,par,step_size=0.001,ctx=model_ctx)

train_sgld=True
num_epochs=100

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                                data_loader=train_data,
                                verbose=True,chain_name='posterior_softmax.h5')

posterior_samples=h5py.File('posterior_softmax.h5','r')
total_samples,total_labels,log_like=inference.predict(posterior_samples,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

loss=posterior_samples.attrs['loss'][:]
plot_loss(loss,'SGLD Softmax','sgld_nonhierarchical_softmax.pdf')

samples={var:posterior_samples[var] for var in posterior_samples.keys()}
samples={var:np.swapaxes(samples[var],0,1) for var in model.par}

rhat = {var:r_hat_estimate(samples[var]) for var in model.par}
plot_diagnostics(rhat,'Rhat','rhat_nonhierarchical_softmax.pdf')
median_rhat_flat={var:np.median(rhat[var]) for var in rhat}
print('Rhat Non-Hierarchical Model',median_rhat_flat)

ess = {var:ess_estimate(samples[var]) for var in model.par}
median_ess_flat={var:np.median(ess[var]) for var in ess}
print('ESS Non-Hierarchical Model',median_ess_flat)
plot_diagnostics(ess,'ESS','ess_nonhierarchical_softmax.pdf')

loo,loos,ks=psisloo(log_like)
max_ks=max(ks[~ np.isinf(ks)])
ks[np.isinf(ks)]=max_ks
flat_ks_1=np.sum(ks>1)
flat_ks_7_1=np.sum(np.logical_and(ks>0.7,ks<1))
flat_ks_5_7=np.sum(np.logical_and(ks>0.5,ks<0.7))
flat_ks_5=np.sum(ks<0.5)
k_loo=1-np.clip(ks,0,1)
print('F_loo : ')
print(classification_report(np.int32(total_labels),np.int32(y_hat),sample_weight=k_loo))
print('#######################################')
print('Hierarchical Stochastic Gradient Langevin Dynamics')


inference=hierarchical_sgld(model,par,step_size=0.001,ctx=model_ctx)

train_sgld=True
num_epochs=100

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                                data_loader=train_data,
                                verbose=True,chain_name='hierarchical_softmax_posterior.h5')

posterior_samples=h5py.File('hierarchical_softmax_posterior.h5','r')
loss=posterior_samples.attrs['loss'][:]
plot_loss(loss,'SGLD Hierarchical Softmax','sgld_hierarchical_softmax.pdf')

total_samples,total_labels,log_like=inference.predict(posterior_samples,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

samples={var:posterior_samples[var] for var in posterior_samples.keys()}
samples={var:np.swapaxes(samples[var],0,1) for var in model.par}

rhat = {var:r_hat_estimate(samples[var]) for var in model.par}
median_rhat_hierarchical={var:np.median(rhat[var]) for var in rhat}
print('Rhat Hierarchical Model',median_rhat_hierarchical)
plot_diagnostics(rhat,'Rhat','rhat_hierarchical_softmax.pdf')


ess_estimate = lambda samples : tfp.mcmc.diagnostic.effective_sample_size(samples, filter_beyond_positive_pairs=False,cross_chain_dims=1).numpy()
ess = {var:ess_estimate(samples[var]) for var in model.par}
median_ess_hierarchical={var:np.median(ess[var]) for var in ess}
print('ESS Hierarchical Model',median_ess_hierarchical)
plot_diagnostics(ess,'ESS','ess_hierarchical_softmax.pdf')


loo,loos,ks=psisloo(log_like)
hierarchical_ks_1=np.sum(ks>1)
hierarchical_ks_7_1=np.sum(np.logical_and(ks>0.7,ks<1))
hierarchical_ks_5_7=np.sum(np.logical_and(ks>0.5,ks<0.7))
hierarchical_ks_5=np.sum(ks<0.5)

k_loo=1-np.clip(ks,0,1)
print('F_loo : ')
print(classification_report(np.int32(total_labels),np.int32(y_hat),sample_weight=k_loo))

import pandas as pd

hierarchical=[hierarchical_ks_1,hierarchical_ks_7_1,hierarchical_ks_5_7,hierarchical_ks_5]
flat=[flat_ks_1,flat_ks_7_1,flat_ks_5_7,flat_ks_5]
index = ['k>1', '0.7<k<1', '0.5<k<0.7','k<0.5']
df = pd.DataFrame({'hierarchical': hierarchical,
                   'non-hierarchical': flat}, index=index)
ax = df.plot.bar(rot=0)
plt.title('Pareto K shape')
plt.savefig('pareto_k_softmax.pdf', bbox_inches='tight')

hierarchical=median_rhat_hierarchical.values()
flat=median_rhat_flat.values()
index = median_rhat_flat.keys()
df = pd.DataFrame({'hierarchical': hierarchical,
                   'non-hierarchical': flat}, index=index)
ax = df.plot.bar(rot=0)
plt.title('Potential Scale Reduction (Rhat)')
plt.savefig('potential_scale_softmax.pdf', bbox_inches='tight')


hierarchical=median_ess_hierarchical.values()
flat=median_ess_flat.values()
index = median_ess_flat.keys()
df = pd.DataFrame({'hierarchical': hierarchical,
                   'non-hierarchical': flat}, index=index)
ax = df.plot.bar(rot=0)
plt.title('ESS') 
plt.savefig('ess_softmax.pdf', bbox_inches='tight')


