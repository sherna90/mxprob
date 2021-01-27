import warnings
warnings.filterwarnings("ignore")

import numpy as np
from hamiltonian.utils import *
from copy import deepcopy
from numpy.linalg import norm
import time

class logistic:
    
    def __init__(self,_hyper):
        self.hyper={var:np.asarray(_hyper[var]) for var in _hyper.keys()}

    def log_prior(self, par,**args):
        K=0
        for var in par.keys():
            dim=(np.asarray(par[var])).size
            K+=dim*0.5*np.log(self.hyper['alpha']/(2*np.pi))
            K-=0.5*self.hyper['alpha']*np.sum(np.square(par[var]))
        return K
    

    def grad(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=np.asarray(v)
                n_data=X.shape[0]
            elif k=='y_train':
                y=np.asarray(v)
        yhat=self.net(par,**args)
        diff = y.reshape(-1,1)-yhat
        #diff=diff[:,:-1]
        grad_w = np.dot(X.T, diff)
        grad_b = np.sum(diff, axis=0)
        grad={}
        grad['weights']=grad_w-self.hyper['alpha']*par['weights']
        grad['weights']=-1.0*grad['weights']
        grad['bias']=grad_b-self.hyper['alpha']*par['bias']
        grad['bias']=-1.0*grad['bias']
        return grad	

    def net(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=np.asarray(v)
        y_linear = np.dot(X, par['weights']) + par['bias']
        y_linear=np.minimum(y_linear,-np.log(np.finfo(float).eps))
        y_linear=np.maximum(y_linear,-np.log(1./np.finfo(float).tiny-1.0))
        yhat = self.sigmoid(y_linear)
        return yhat

    def sigmoid(self, y_linear):
        norms=(1.0 + np.exp(-y_linear))
        return 1.0 / norms

    def negative_log_posterior(self, par,**args ):
        for k,v in args.items():
            if k=='X_train':
                X=np.asarray(v)
                n_data=X.shape[0]
        return (-1.0/n_data)*(self.log_likelihood(par,**args)+self.log_prior(par,**args))

    def log_likelihood(self, par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=np.asarray(v)
            elif k=='y_train':
                y=np.asarray(v)
        y_pred=np.squeeze(self.net(par,**args),axis=1)
        ll = np.sum(np.multiply(y,np.log(y_pred))+np.multiply((1.0-y),np.log(1.0-y_pred)))
        return ll


    def predict(self, par,X,prob=False,batchsize=32):
        results=[]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            X_batch=X[excerpt] 
            yhat=self.net(par,X_train=X_batch)
            if prob:
                out=yhat
            else:
                out=(yhat>0.5).astype(int).flatten()
            results.append(out)
        results=np.asarray(results)
        return results.flatten()	


