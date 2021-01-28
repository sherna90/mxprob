import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon


class softmax():
    
    def __init__(self,_hyper,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.LOG2PI = np.log(2.0 * np.pi)

    def cross_entropy(self, y_hat, y):
        return -nd.nansum(y * nd.log_softmax(y_hat), axis=0, exclude=True)

    def softmax(self, y_linear):
        exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
        norms = nd.sum(exp, axis=1).reshape((-1,1))
        return exp / norms

    def predict(self, par,X,prob=False):
        yhat=self.forward(par,X_train=X)   
        if prob:
            out=yhat
        else:
            pred=yhat.argmax(axis=1)
            out=pred
        return out	

    def forward(self,par, **args):
        for k,v in args.items():
            if k=='X_train':
                X=v
        y_linear = nd.dot(X, par['weights']) + par['bias']
        #y_linear=np.minimum(y_linear,-np.log(np.finfo(float).eps))
        #y_linear=np.maximum(y_linear,-np.log(1./np.finfo(float).tiny-1.0))
        yhat = self.softmax(y_linear)
        return yhat

    def grad(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v
            elif k=='y_train':
                y=v
        yhat=self.forward(par,X_train=X)
        diff = y-yhat
        grad_w = nd.dot(X.T, diff)
        grad_b = nd.sum(diff, axis=0)
        grad={}
        grad['weights']=grad_w
        grad['weights']=-1.0*grad['weights']
        grad['bias']=grad_b
        grad['bias']=-1.0*grad['bias']
        return grad	
     
    def negative_log_prior(self, par,**args):
        log_gaussian= lambda x,mu,sigma :  -0.5 * self.LOG2PI - nd.log(sigma) - (x - mu) ** 2 / (2 * sigma ** 2)
        K=0
        for var in par.keys():
            K+=nd.sum(log_gaussian(par[var], 0., self.hyper['alpha']))
        return K
    
    def negative_log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v
            elif k=='y_train':
                y=v
        y_hat = self.forward(par,X_train=X)
        return 1.0*nd.sum(self.cross_entropy(y_hat,y))
        
    def negative_log_posterior(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v
                n_data=X.shape[0]
        return (1.0/n_data)*(self.negative_log_likelihood(par,**args))
        #+self.negative_log_prior(par,**args))