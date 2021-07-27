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
# In[3]:


transform = transforms.Compose([
    transforms.Resize([150,150]),
    transforms.ToTensor()
])


# In[4]:


training_path='~/code/PlantVillage-Dataset/data_distribution_for_SVM/train'
testing_path='~/code/PlantVillage-Dataset/data_distribution_for_SVM/test'


# In[5]:


num_gpus = 1
model_ctx = mx.gpu()

num_workers = 2
batch_size = 32 

train_data = mx.gluon.data.vision.datasets.ImageFolderDataset(training_path).transform_first(transform)
test_data = mx.gluon.data.vision.datasets.ImageFolderDataset(testing_path).transform_first(transform)


# In[6]:


train_data_loader = mx.gluon.data.DataLoader(train_data, batch_size, shuffle=True, num_workers=num_workers)
val_data_loader = mx.gluon.data.DataLoader(test_data, batch_size, num_workers=num_workers)



# ### Bayesian inference for Plant Village
# 
# * [Stochastic Gradient Descent](#chapter1)
# * [Stochastic Gradient Langevin Dynamics](#chapter2)
# * [Bayes By Backprop](#chapter3)
# * [Diagnostics](#chapter4)
# 

# # Stochastic Gradient Descent <a class="anchor" id="chapter1"></a>

# In[9]:


import mxnet as mx
from mxnet import nd, autograd, gluon
hyper={'alpha':10.}
in_units=(3,150,150)
out_units=38


# In[10]:


import mxnet as mx
from hamiltonian.inference.sgd import sgd
from hamiltonian.models.softmax import vgg_softmax

model=vgg_softmax(hyper,in_units,out_units,n_layers=16,ctx=model_ctx)
inference=sgd(model,model.par,step_size=0.001,ctx=model_ctx)



train_sgd=True
num_epochs=100

if train_sgd:
    par,loss=inference.fit(epochs=num_epochs,batch_size=batch_size,data_loader=train_data_loader,verbose=True)

    fig=plt.figure(figsize=[5,5])
    plt.plot(loss,color='blue',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGD VGG Plant Village', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgd_vgg.pdf', bbox_inches='tight')
    model.net.save_parameters('results/vgg/vgg_sgd_'+str(num_epochs)+'_epochs.params')
else:
    model.net.load_parameters('results/vgg/vgg_sgd_'+str(num_epochs)+'_epochs.params',ctx=model_ctx)
    par=dict()
    for name,gluon_par in model.net.collect_params().items():
        par.update({name:gluon_par.data()})
    


# In[16]:


model.net.collect_params()


# In[17]:


total_samples,total_labels,log_like=inference.predict(par,batch_size=batch_size,num_samples=100,data_loader=val_data_loader)


# In[18]:


y_hat=np.quantile(total_samples,.9,axis=0)


# In[19]:


from sklearn.metrics import classification_report

print(classification_report(np.int32(total_labels),np.int32(y_hat)))


# # Stochastic Gradient Langevin Dynamics <a class="anchor" id="chapter2"></a>

# In[28]:


from hamiltonian.inference.sgld import sgld
from hamiltonian.models.softmax import vgg_softmax

model=vgg_softmax(hyper,in_units,out_units,n_layers=16,ctx=model_ctx)
inference=sgld(model,model.par,step_size=0.001,ctx=model_ctx)



import matplotlib.pyplot as plt
import seaborn as sns
import glob

train_sgld=True
num_epochs=250

if train_sgld:
    loss,posterior_samples=inference.sample(epochs=num_epochs,batch_size=batch_size,
                                data_loader=train_data_loader,
                                verbose=True,chain_name='chain_nonhierarchical')

    plt.rcParams['figure.dpi'] = 360
    sns.set_style("whitegrid")
    fig=plt.figure(figsize=[5,5])
    plt.plot(loss[0],color='blue',lw=3)
    plt.plot(loss[1],color='red',lw=3)
    plt.xlabel('Epoch', size=18)
    plt.ylabel('Loss', size=18)
    plt.title('SGLD VGG Plant Village', size=18)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.savefig('sgld_vgg.pdf', bbox_inches='tight')
else:
    chain1=glob.glob("results/vgg/chain_nonhierarchical_0_1_sgld*")
    chain2=glob.glob("results/vgg/chain_nonhierarchical_0_sgld*")
    posterior_samples=[chain1,chain2]


# In[31]:


posterior_samples_flat=[item for sublist in posterior_samples for item in sublist]


# In[32]:


total_samples,total_labels,log_like=inference.predict(posterior_samples_flat,5,data_loader=val_data_loader)


# In[33]:


from sklearn.metrics import classification_report
posterior_samples

y_hat=np.quantile(total_samples,.9,axis=0)

print(classification_report(np.int32(total_labels),np.int32(y_hat)))


# In[42]:


from sklearn.metrics import f1_score

score=[]
for q in np.arange(.1,.9,.1):
    y_hat=np.quantile(total_samples,q,axis=0)
    score.append(f1_score(np.int32(total_labels),np.int32(y_hat), average='macro'))
print('mean f-1 : {0}, std f-1 : {1}'.format(np.mean(score),2*np.std(score)))


# In[34]:


import arviz as az

posterior_samples_multiple_chains=inference.posterior_diagnostics(posterior_samples)
datasets=[az.convert_to_inference_data(sample) for sample in posterior_samples_multiple_chains]
dataset = az.concat(datasets, dim="chain")
mean_r_hat_values={var:float(az.rhat(dataset)[var].mean().data) for var in model.par}
mean_ess_values={var:float(az.ess(dataset)[var].mean().data) for var in model.par}
mean_mcse_values={var:float(az.mcse(dataset)[var].mean().data) for var in model.par}


# In[35]:


az.summary(dataset)


# In[36]:


print(mean_r_hat_values)


# In[37]:


print(mean_ess_values)


# In[38]:


print(mean_mcse_values)


# In[39]:


from hamiltonian.psis import *

loo,loos,ks=psisloo(-log_like)


# In[41]:


from sklearn.metrics import f1_score

score=[]
for q in np.arange(.1,.9,.1):
    y_hat=np.quantile(total_samples,q,axis=0)
    score.append(f1_score(np.int32(total_labels),np.int32(y_hat), sample_weight=loos,average='weighted'))
print('mean f-1 : {0}, std f-1 : {1}'.format(np.mean(score),2*np.std(score)))


# In[ ]:




