import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp

class linear():
    
    def __init__(self,_hyper,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx

    def predict(self, par,X,prob=False):
        yhat=self.forward(par,X_train=X)    
        return yhat	

    def forward(self,par, **args):
        for k,v in args.items():
            if k=='X_train':
                X=nd.array(v,ctx=self.ctx)
        y_linear = nd.dot(X, par['weights']) + par['bias']
        y_hat=mxp.normal.Normal(loc=y_linear,scale=np.sqrt(self.hyper['scale']))
        return y_hat
     
    def negative_log_prior(self, par,**args):
        log_prior=nd.zeros(1)
        for var in par.keys():
            means=nd.zeros(par[var].shape,ctx=self.ctx)
            sigmas=nd.ones(par[var].shape,ctx=self.ctx)*np.sqrt(self.hyper['alpha'])
            param_prior=mxp.normal.Normal(loc=means,scale=sigmas)
            log_prior=log_prior-nd.mean(param_prior.log_prob(par[var]).as_nd_ndarray())
        return log_prior
       
    def negative_log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=nd.array(v,ctx=self.ctx)
            elif k=='y_train':
                y=nd.array(v,ctx=self.ctx)
        y_hat = self.forward(par,X_train=X)
        return -nd.mean(y_hat.log_prob(y).as_nd_ndarray())
        
    def loss(self,par,**args):
        log_like=self.negative_log_likelihood(par,**args)
        log_prior=self.negative_log_prior(par,**args)
        return log_like+log_prior

class linear_aleatoric(linear):

    def forward(self,par, **args):
        softplus = lambda x : nd.log(1. + nd.exp(x))
        for k,v in args.items():
            if k=='X_train':
                X=nd.array(v,ctx=self.ctx)
        y_mean = nd.dot(X, par['weights']) + par['bias'] 
        y_scale = softplus(nd.dot(X, par['weights_scale']) + par['bias_scale'])
        #y_scale=nd.where(y_scale>30,y_scale,nd.log1p(nd.exp(y_scale)))
        y_hat=mxp.normal.Normal(loc=y_mean,scale=y_scale)
        return y_hat

     
    