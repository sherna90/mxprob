import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv,norm
from copy import deepcopy
from multiprocessing import Pool,cpu_count
import os 

from tqdm import tqdm, trange
import h5py 
import time

class sgmcmc:

    def __init__(self,model, start_p, path_length=1.0,step_size=0.1,verbose=True):
        self.start={var:np.asarray(start_p[var]) for var in start_p.keys()}
        self.step_size = step_size
        self.path_length = path_length
        self.model = model
        self.verbose=verbose
        self.mu = np.log(10 * self.step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = 0.65
        self.gamma = 0.05
        self.t0 = 10.0
        self.t = 0
        self.kappa = 0.75
        self.error_sum = 0.0
        self.log_averaged_step = 0.0

    def step(self,state,momentum,rng,**args):
        pass

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield X[excerpt], y[excerpt]

    def sample(self,epochs=1,burnin=1,batch_size=1,rng=None,**args):
        if rng == None:
            rng = np.random.RandomState()
        X=args['X_train']
        y=args['y_train']
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        num_batches=np.ceil(y[:].shape[0]/float(batch_size))
        decay_factor=self.step_size/num_batches
        #q,p=self.start,self.draw_momentum(rng)
        q,p=self.start,{var:np.zeros_like(self.start[var]) for var in self.start.keys()}
        print('start burnin')
        for i in tqdm(range(int(burnin))):
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                kwargs={'X_train':X_batch,'y_train':y_batch,'verbose':verbose}
                q,p=self.step(q,p,rng,**kwargs)
                if (j % 10)==0:
                    ll=-1.0*self.model.log_likelihood(q,**kwargs)
                    print('burnin {0}, loss: {1:.4f}, mini-batch update : {2}'.format(i,ll,j))
                j=j+1
        logp_samples=np.zeros(epochs)
        posterior={var:[] for var in self.start.keys()}
        print('start sampling')
        initial_step_size=self.step_size
        for i in tqdm(range(epochs)):
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y, batch_size):
                kwargs={'X_train':X_batch,'y_train':y_batch,'verbose':verbose}
                q,p=self.step(q,p,rng,**kwargs)
                self.step_size=self.lr_schedule(initial_step_size,j,decay_factor,num_batches)
                if (j % 10)==0:
                    ll=-1.0*self.model.log_likelihood(q,**kwargs)
                    print('epoch {0}, loss: {1:.4f}, mini-batch update : {2}'.format(i,ll,j))
                j=j+1
            #initial_step_size=self.step_size
            logp_samples[i]=self.model.negative_log_posterior(q,**kwargs)
            for var in self.start.keys():
                posterior[var].append(q[var])
            if self.verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(logp_samples[i]))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior,logp_samples

    def lr_schedule(self,initial_step_size,step,decay_factor,num_batches):
        return initial_step_size * (1.0/(1.0+step*decay_factor*num_batches))