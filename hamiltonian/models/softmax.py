import warnings
warnings.filterwarnings("ignore")

import numpy as np
from hamiltonian.utils import *
from copy import deepcopy
from numpy.linalg import norm
from scipy.special import logsumexp
import time
from tqdm import tqdm

class softmax:
    
    def __init__(self,_hyper):
        self.hyper=_hyper

    def cross_entropy(self, y_linear, y):
        lse=logsumexp(y_linear,axis=1)
        y_hat=y_linear-np.repeat(lse[:,np.newaxis],y.shape[1]).reshape(y.shape)
        return np.sum(y *  y_hat,axis=1)

    def log_prior(self, par,**args):
        for k,v in args.items():
            if k=='y_train':
                y=v
        K=0
        for var in par.keys():
            dim=(np.array(par[var])).size
            K-=0.5*dim*np.log(2*np.pi)-0.5*dim*np.log(self.hyper['alpha'])
        return K

    def softmax(self, y_linear):
        #y_linear=np.hstack((y_linear,np.zeros((y_linear.shape[0],1))))
        exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
        norms = np.sum(exp, axis=1).reshape((-1,1))
        return exp / norms

    def net(self,par, X):
        y_linear = np.dot(X, par['weights']) + par['bias']
        y_linear=np.minimum(y_linear,-np.log(np.finfo(float).eps))
        y_linear=np.maximum(y_linear,-np.log(1./np.finfo(float).tiny-1.0))
        yhat = self.softmax(y_linear)
        return yhat

    def grad(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v
            elif k=='y_train':
                y=v
        yhat=self.net(par,X)
        diff = y-yhat
        #diff=diff[:,:-1]
        grad_w = np.dot(X.T, diff)
        grad_b = np.sum(diff, axis=0)
        grad={}
        grad['weights']=grad_w-self.hyper['alpha']*par['weights']
        grad['weights']=-1.0*grad['weights']
        grad['bias']=grad_b-self.hyper['alpha']*par['bias']
        grad['bias']=-1.0*grad['bias']
        return grad	
    
    def log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v
            elif k=='y_train':
                y=v
        y_linear = np.dot(X, par['weights']) + par['bias']
        y_linear=np.minimum(y_linear,-np.log(np.finfo(float).eps))
        y_linear=np.maximum(y_linear,-np.log(1./np.finfo(float).tiny-1.0))
        return np.sum(self.cross_entropy(y_linear,y))
        
    def negative_log_posterior(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=np.asarray(v)
                n_data=X.shape[0]
        return (-1.0/n_data)*(self.log_likelihood(par,**args)+self.log_prior(par,**args))


    def predict(self, par,X,prob=False):
        yhat=self.net(par,X)   
        if prob:
            out=yhat
        else:
            pred=yhat.argmax(axis=1)
            out=pred
        return out	

    def predict_stochastic(self,par,X,prob=False,p=0.5):
        n_x,n_y=X.shape
        Z=np.random.binomial(1,p,size=X.shape)
        X_t=np.multiply(X,Z)   
        yhat=self.net(par,X_t)
        if prob:
            out=yhat
        else:
            out=yhat.argmax(axis=1)
        return out	
