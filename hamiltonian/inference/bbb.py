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
    def loss(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X_batch=v
            elif k=='y_train':
                y_batch=v
            elif k=='stds':
                stds=v
            elif k=='e':
                e=v
        mean_samples={}
        std_samples={}
        l_kl=0
        for var in par.keys():
            mean_samples.update({var:par[var]}) 
            std_samples.update({var:self.softplus(stds[var])})
        for var in par.keys():
            l_kl =l_kl+(1.0 + nd.log(std_samples[var]**2) - std_samples[var]**2 - mean_samples[var]**2)
            par.update({var:mean_samples[var] + (std_samples[var] * (e[var].sample(1).as_nd_ndarray()))}) 
            #par.update({var:mean_samples[var]}) 
        l_kl=-0.5*l_kl
        # mu_a = par['weights']
        # sig_a = self.softplus(par['weights_std'])


        # mu_b = par['bias']
        # sig_b = self.softplus(par['bias_std'])

        # mu_c = par['weights_scale']
        # sig_c = nd.log1p(nd.exp(par['weights_scale_std']))#self.sofutpls()

        # mu_d = par['bias_scale']
        # sig_d = nd.log1p(nd.exp(par['bias_scale_std']))#self.softplus(par['bias_scale_std'])

        #divergencia kl
        # l_kl = -0.5*(1.0 + nd.log(sig_a**2) - sig_a**2 - mu_a**2 
        #         + 1.0 + nd.log(sig_b**2) - sig_b**2 - mu_b**2+
        #         + 1.0 + nd.log(sig_c**2) - sig_c**2 - mu_c**2
        #         + 1.0 + nd.log(sig_d**2) - sig_d**2 - mu_d**2)

        #muestras a y b
        # a = mu_a + (sig_a * (e['weights'].sample(1).as_nd_ndarray()))
        # #a = a.asscalar()
        # b = mu_b + (sig_b * (e['bias'].sample(1).as_nd_ndarray()))
        # #b = b.asscalar()
        # c = mu_c + (sig_c * (e['weights_scale'].sample(1).as_nd_ndarray()))
        # #a = a.asscalar()
        # d = mu_d + (sig_d * (e['bias_scale'].sample(1).as_nd_ndarray()))
        # #b = b.asscalar()     
        
        #forward(linear)
        #X = nd.array(X_batch,ctx=self.ctx)
        #y_linear = nd.dot(X,a) + b
        #y_prob = mxp.normal.Normal(loc=y_linear,scale=self.sigma)
        
        
        #likelihood
        #par2={'weights':a,'bias':b,'weights_scale':c,'bias_scale':nd.array([0.001])}
        y = nd.array(y_batch,ctx=self.ctx)
        l_nll = self.model.loss(par,X_train=X_batch,y_train=y_batch)
        #-nd.mean(y_prob.log_prob(y).as_nd_ndarray())
        
        
        #loss function
        loss = l_kl/(y_batch.shape[0]-1) + l_nll 
        
        return loss
    
    def step(self,batch_size,momentum,par):
        momentum, par = sgd.step(self, batch_size,momentum,par)
        return momentum, par
