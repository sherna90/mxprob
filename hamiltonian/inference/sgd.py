import numpy as np
import scipy as sp
import os
from numpy.linalg import inv,norm
from copy import deepcopy
import os 
from tqdm import tqdm, trange
import h5py 
import time

class sgd:
    
    def __init__(self,model,start_p,step_size=0.1):
        self.start = start_p
        self.step_size = step_size
        self.model = model
        

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]

    def fit(self,epochs=1,batch_size=1,gamma=0.9,**args):
        X=args['X_train']
        y=args['y_train']
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        par=deepcopy(self.start)
        momentum={var:np.zeros_like(par[var]) for var in par.keys()}
        for i in tqdm(range(epochs)):
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                grad_p=self.model.grad(par,X_train=X_batch,y_train=y_batch)
                for var in par.keys():
                    momentum[var] = gamma * momentum[var] - self.step_size * grad_p[var]
                    par[var]+=momentum[var]
            loss_val[i]=self.model.negative_log_posterior(par,X_train=X_batch,y_train=y_batch)
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val

    def fit_dropout(self,epochs=1,batch_size=1,gamma=0.9,p=0.5,**args):
        X=args['X_train']
        y=args['y_train']
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        par=deepcopy(self.start)
        momentum={var:np.zeros_like(par[var]) for var in par.keys()}
        for i in tqdm(range(epochs)):
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                n_batch=np.float(y_batch.shape[0])
                Z=np.random.binomial(1,p,size=X_batch.shape)
                X_batch_dropout=np.multiply(X_batch,Z)
                grad_p=self.model.grad(par,X_train=X_batch_dropout,y_train=y_batch)
                for var in par.keys():
                    momentum[var] = gamma * momentum[var] - self.step_size * grad_p[var]
                    par[var]+=momentum[var]
            loss_val[i]=-1.*self.model.log_likelihood(par,X_train=X_batch,y_train=y_batch)
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val