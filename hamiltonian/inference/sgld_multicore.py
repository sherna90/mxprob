import numpy as np
import scipy as sp
import math
import os
from hamiltonian.utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count,Process,Queue
import os 
from hamiltonian.cpu.sgld import sgld

from tqdm import tqdm, trange
import h5py 
import time

def unwrap_self_sgld(arg, **kwarg):
    return sgld_multicore.sample(*arg, **kwarg)

class sgld_multicore(sgld):

    def single_epoch(self,q,momentum,n_data,batch_size,rng):
        for j in range(int(n_data/batch_size)):
            X_batch, y_batch = sgld_multicore.sample.queue.get()
            q,momentum=self.step(momentum,X_batch,y_batch,q,rng)
            if (j%1000 == 0):
                iter_loss=-1.0*self.model.log_likelihood(X_batch,y_batch,q,self.hyper)
                print('core : {0}, minibatch : {1}, loss: {2:.4f}'.format(os.getpid(),j,iter_loss))
        final_loss=-1.0*self.model.log_likelihood(X_batch,y_batch,q,self.hyper)
        return q,momentum,final_loss

    def sample(self,epochs=1e4,burnin=1e3,batch_size=20,backend=None,rng=None):
        q=self.start
        n_data=sgld_multicore.sample.X_train.shape[0]
        momentum={var:np.zeros_like(self.start[var]) for var in self.start.keys()}
        for i in tqdm(range(int(burnin)),total=int(burnin)):
            q,momentum,final_loss=self.single_epoch(q,momentum,n_data,batch_size,rng)
        loss_val=np.zeros(int(epochs))
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=np.float32)
            momentum={var:np.zeros_like(self.start[var]) for var in self.start.keys()}
            for i in tqdm(range(int(epochs)),total=int(epochs)):
                q,momentum,final_loss=self.single_epoch(q,momentum,n_data,batch_size,rng)
                loss_val[i] = final_loss
                if (i % (epochs/10)==0):
                    print('core : {0}, minibatch : {1}, loss: {2:.4f}'.format(os.getpid(),i,loss_val[i]))
                for var in self.start.keys():
                    param_shape=self.start[var].shape
                    posterior[var][-1,:]=q[var]
                    if i<int(epochs-1):
                        posterior[var].resize((posterior[var].shape[0]+1,)+param_shape)
                backend_samples.flush()
            backend_samples.close()
            return backend_samples, loss_val
        else:
            posterior={var:[] for var in self.start.keys()}
            momentum={var:np.zeros_like(self.start[var]) for var in self.start.keys()}
            for i in tqdm(range(int(epochs)),total=int(epochs)):
                q,momentum,final_loss=self.single_epoch(q,momentum,n_data,batch_size,rng)
                loss_val[i] = final_loss
                if (i % (epochs/10)==0):
                    print('loss: {0:.4f}'.format(loss_val[i]))
                for var in self.start.keys():
                    posterior[var].append(q[var])
            for var in self.start.keys():
                print(posterior[var][-1].shape)
                posterior[var]=np.array(posterior[var])
            return posterior, loss_val

    def iterate_minibatches(self, X_train,y_train,queue, batch_size, total):
        for i in range(int(total)):
                #assert X_train.shape[0] == y_train.shape[0]
            for start_idx in range(0, X_train.shape[0] - batch_size + 1, batch_size):
                excerpt = slice(start_idx, start_idx + batch_size)
                queue.put((X_train[excerpt], y_train[excerpt]))

    def sample_init(self, _queue,_X_train,_y_train):
        sgld_multicore.sample.queue = _queue
        sgld_multicore.sample.X_train = _X_train
        sgld_multicore.sample.y_train = _y_train
    
    def multicore_sample(self,X_train,y_train,epochs=1e4,burnin=1e3,batch_size=20,backend=None,ncores=cpu_count()):
        if backend:
            multi_backend = [backend+"_%i.h5" %i for i in range(ncores)]
        else:
            multi_backend = [backend]*ncores    
        rng = [np.random.RandomState(i) for i in range(ncores)]
        queue = Queue(maxsize=ncores)    
        multicore_epochs=max(int(epochs/ncores),1)  
        l = Process(target=sgld_multicore.iterate_minibatches, args=(self,X_train, y_train,queue, batch_size, ( int(epochs) + int(burnin)*ncores)))
        p = Pool(None, sgld_multicore.sample_init, [self, queue,X_train,y_train])
        l.start()
        results=p.map(unwrap_self_sgld, zip([self]*ncores,[int(multicore_epochs)]*ncores,[int(burnin)]*ncores,[batch_size]*ncores, multi_backend,rng))
        l.join()
        if not backend:
            posterior={var:np.concatenate([results[i][0][var] for i in range(len(results))],axis=0) for var in self.start.keys()}
            logp_samples=np.concatenate([results[i][1] for i in range(len(results))],axis=0)
            return posterior,logp_samples 
        else:
            logp_samples=np.concatenate([results[i][1] for i in range(len(results))],axis=0)
            return multi_backend, logp_samples