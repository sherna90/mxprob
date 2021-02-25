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
                X=v
        y_linear = nd.dot(X, par['weights']) + par['bias']
        y_hat=mxp.normal.Normal(loc=y_linear,scale=1)
        return y_hat

     
    def negative_log_prior(self, par,**args):
        K=nd.zeros(1)
        for var in par.keys():
            prior=mxp.normal.Normal(loc=0.,scale=self.hyper['alpha'])
            K=K-prior.log_prob(par[var])
        return K
       
    def negative_log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v
            elif k=='y_train':
                y=v
        y_hat = self.forward(par,X_train=X)
        return -nd.mean(y_hat.log_prob(y))
        
    def negative_log_posterior(self,par,**args):
        return self.negative_log_likelihood(par,**args)