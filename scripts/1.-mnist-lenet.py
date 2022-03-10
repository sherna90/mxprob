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
from hamiltonian.models.softmax import softmax,lenet
from hamiltonian.inference.sgld import sgld
from hamiltonian.inference.sgld import hierarchical_sgld,distillation_sgld
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
num_workers = 0
batch_size = 512 
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
inference=sgd(model,step_size=0.001,ctx=model_ctx)

train_sgd=False
num_epochs=50
if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,
                           data_loader=train_data,chain_name='lenet_map.h5',verbose=True)
map_estimate=h5py.File('lenet_map.h5','r')
par={var:map_estimate[var][:] for var in map_estimate.keys()}
params=model.net.collect_params()
[params[var].set_data(map_estimate[var][:]) for var in params.keys()]
map_estimate.close()

total_samples,total_labels,log_like=inference.predict(par,batch_size=batch_size,num_samples=100,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

''''
print('#######################################')
print('Stochastic Gradient Langevin Dynamics')
student=lenet(hyper,in_units,out_units,ctx=model_ctx)
inference=distillation_sgld(student,step_size=1e-2,ctx=model_ctx)

train_sgld=False
num_epochs=100

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                                data_loader=train_data,
                                verbose=True,chain_name='lenet_posterior_ditillation.h5',
                                teacher=model)

posterior_samples=h5py.File('lenet_posterior_distillation.h5','r')
total_samples,total_labels,log_like=inference.predict(posterior_samples,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))


samples={var:posterior_samples[var] for var in posterior_samples.keys()}
samples={var:np.swapaxes(samples[var],0,1) for var in posterior_samples.keys()}
r_hat_estimate = lambda samples : tfp.mcmc.diagnostic.potential_scale_reduction(samples, independent_chain_ndims=1,split_chains=False).numpy()
rhat = {var:r_hat_estimate(samples[var]) for var in posterior_samples.keys()}

labels, data = rhat.keys(), rhat.values()
flatten_data=list()
for d in data:
    flatten_data.append(d.reshape(-1))

plt.rcParams['figure.dpi'] = 360
sns.set_style("whitegrid")
plt.boxplot(flatten_data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.title('Rhat')
plt.savefig('rhat_distilled_lenet.pdf', bbox_inches='tight')

median_rhat_flat={var:np.median(rhat[var]) for var in rhat}

print('Rhat Distilled Lenet',median_rhat_flat)

ess_estimate = lambda samples : tfp.mcmc.diagnostic.effective_sample_size(samples, filter_beyond_positive_pairs=False,cross_chain_dims=1).numpy()
ess = {var:ess_estimate(samples[var]) for var in posterior_samples.keys()}

median_ess_flat={var:np.median(ess[var]) for var in ess}
print('ESS Distilled Lenet',median_ess_flat)

labels, data = ess.keys(), ess.values()
flatten_data=list()
for d in data:
    flatten_data.append(d.reshape(-1))
plt.rcParams['figure.dpi'] = 360
sns.set_style("whitegrid")    
plt.boxplot(flatten_data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.title('ESS')
plt.savefig('ess_distilled_lenet.pdf', bbox_inches='tight')

loo,loos,ks=psisloo(log_like)
#max_ks=max(ks[~ np.isinf(ks)])
#ks[np.isinf(ks)]=max_ks
flat_ks_1=np.sum(ks>1)
flat_ks_7_1=np.sum(np.logical_and(ks>0.7,ks<1))
flat_ks_5_7=np.sum(np.logical_and(ks>0.5,ks<0.7))
flat_ks_5=np.sum(ks<0.5)

'''
print('#######################################')
print('Hierarchical Stochastic Gradient Langevin Dynamics')


inference=sgld(model,step_size=0.001,ctx=model_ctx)

train_sgld=True
num_epochs=100

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                                data_loader=train_data,
                                verbose=True,chain_name='lenet_posterior.h5')



posterior_samples=h5py.File('lenet_posterior.h5','r')
total_samples,total_labels,log_like=inference.predict(posterior_samples,data_loader=val_data)
y_hat=np.quantile(total_samples,.5,axis=0)
print(classification_report(np.int32(total_labels),np.int32(y_hat)))

samples={var:posterior_samples[var] for var in posterior_samples.keys()}
samples={var:np.swapaxes(samples[var],0,1) for var in posterior_samples.keys()}
r_hat_estimate = lambda samples : tfp.mcmc.diagnostic.potential_scale_reduction(samples, independent_chain_ndims=1,split_chains=False).numpy()
rhat = {var:r_hat_estimate(samples[var]) for var in posterior_samples.keys()}

median_rhat_hierarchical={var:np.median(rhat[var]) for var in rhat}
median_rhat_hierarchical

print('Rhat Lenet',median_rhat_hierarchical)

labels, data = rhat.keys(), rhat.values()
flatten_data=list()
for d in data:
    flatten_data.append(d.reshape(-1))

plt.rcParams['figure.dpi'] = 360
sns.set_style("whitegrid")
plt.boxplot(flatten_data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.title('Rhat')
plt.savefig('rhar_lenet.pdf', bbox_inches='tight')


ess_estimate = lambda samples : tfp.mcmc.diagnostic.effective_sample_size(samples, filter_beyond_positive_pairs=False,cross_chain_dims=1).numpy()
ess = {var:ess_estimate(samples[var]) for var in posterior_samples.keys()}
median_ess_hierarchical={var:np.median(ess[var]) for var in ess}

print('ESS Lenet',median_ess_hierarchical)

labels, data = ess.keys(), ess.values()
flatten_data=list()
for d in data:
    flatten_data.append(d.reshape(-1))

plt.rcParams['figure.dpi'] = 360
sns.set_style("whitegrid")
plt.boxplot(flatten_data)
plt.xticks(range(1, len(labels) + 1), labels)
plt.title('ESS')
plt.savefig('ess_lenet.pdf', bbox_inches='tight')


loo,loos,ks=psisloo(log_like)
#max_ks=max(ks[~ np.isinf(ks)])
#ks[np.isinf(ks)]=max_ks
hierarchical_ks_1=np.sum(ks>1)
hierarchical_ks_7_1=np.sum(np.logical_and(ks>0.7,ks<1))
hierarchical_ks_5_7=np.sum(np.logical_and(ks>0.5,ks<0.7))
hierarchical_ks_5=np.sum(ks<0.5)

'''
import pandas as pd

hierarchical=[hierarchical_ks_1,hierarchical_ks_7_1,hierarchical_ks_5_7,hierarchical_ks_5]
flat=[flat_ks_1,flat_ks_7_1,flat_ks_5_7,flat_ks_5]
index = ['k>1', '0.7<k<1', '0.5<k<0.7','k<0.5']
df = pd.DataFrame({'sgld': hierarchical,
                   'distilled_sgld': flat}, index=index)
ax = df.plot.bar(rot=0)
plt.title('Pareto K shape')
plt.savefig('pareto_k_lenet.pdf', bbox_inches='tight')

import pandas as pd
hierarchical=median_rhat_hierarchical.values()
flat=median_rhat_flat.values()
index = median_rhat_flat.keys()
df = pd.DataFrame({'sgld': hierarchical,
                   'distilled_sgld': flat}, index=index)
ax = df.plot.bar(rot=0)
plt.title('Potential Scale Reduction (Rhat)')
plt.savefig('potential_scale_lenet.pdf', bbox_inches='tight')


hierarchical=median_ess_hierarchical.values()
flat=median_ess_flat.values()
index = median_ess_flat.keys()
df = pd.DataFrame({'sgld': hierarchical,
                   'distilled-sgld': flat}, index=index)
ax = df.plot.bar(rot=0)
plt.title('ESS') 
plt.savefig('ess_lenet.pdf', bbox_inches='tight')

'''
