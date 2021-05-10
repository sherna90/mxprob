import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base
from hamiltonian.inference.sgd import sgd
import mxnet.gluon.probability as mxp

class bbb(base):

    def softplus(self,x):
        return nd.log(1. + nd.exp(x))


    #loss: Bayesian inference
    def floss(self,par, X_batch, y_batch,w_0,e):
        mu_a = par['weights']
        sig_a = self.softplus(par['weights_scale'])


        mu_b = par['bias']
        sig_b = self.softplus(par['bias_scale'])

        #divergencia kl
        l_kl = -0.5*(1.0 + nd.log(sig_a**2) - sig_a**2 - mu_a**2 + 1.0 + nd.log(sig_b**2) - sig_b**2 - mu_b**2)

        #muestras a y b
        a = mu_a + (sig_a * (e.sample(1).as_nd_ndarray()))
        #a = a.asscalar()
        b = mu_b + (sig_b * (e.sample(1).as_nd_ndarray()))
        #b = b.asscalar()
        
        
        #forward(linear)
        X = nd.array(X_batch,ctx=self.ctx)
        y_linear = nd.dot(X,a) + b
        y_prob = mxp.normal.Normal(loc=y_linear,scale=self.sigma)
        
        
        #likelihood
        par2={'weights':a,'bias':b,'weights_scale':par['weights_scale'],'bias_scale':par['bias_scale']}
        y = nd.array(y_batch,ctx=self.ctx)
        l_nll = self.model.negative_log_likelihood(par2,X_train=X_batch,y_train=y_batch)
        #-nd.mean(y_prob.log_prob(y).as_nd_ndarray())
        
        
        #loss function
        loss = l_kl + l_nll 
        
        return loss
    
    def step(self,batch_size,momentum,par):
        momentum, par = sgd.step(self, batch_size,momentum,par)
        return momentum, par
