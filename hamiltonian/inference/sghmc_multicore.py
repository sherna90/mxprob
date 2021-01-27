import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count,Process,Queue
import os 

from hamiltonian.cpu.sghmc import sghmc

from tqdm import tqdm, trange
import h5py 
import time

def unwrap_self_sghmc(arg, **kwarg):
    return sghmc_multicore.sample(*arg, **kwarg)

class sghmc_multicore(sghmc):

    def sample(self,X_train,y_train,niter=1e4,burnin=1e3,batch_size=20,backend=None,rng=None):
        accepted=[]
        rng = np.random.RandomState()
        q,p=self.start,self.draw_momentum(rng)
        self.find_reasonable_epsilon(X_train,y_train,q,rng)
        for i in tqdm(range(int(burnin)),total=int(burnin)):
            for auxiliar in range(len(range(0, X_train.shape[0] - batch_size + 1, batch_size))):
                X_batch, y_batch = sghmc_multicore.sample.queue.get()
                q,p,a=self.step(X_batch,y_batch,q,p,rng)
                accepted.append(a)
        if self._verbose:
            print('burn-in acceptance rate : {0:.4f}'.format(self.acceptance_rate(accepted)))  
        
        del accepted[:]
        logp_samples=np.zeros(int(niter))
        if backend:
            backend_samples=h5py.File(backend)
            posterior={}
            for var in self.start.keys():
                param_shape=self.start[var].shape
                posterior[var]=backend_samples.create_dataset(var,(1,)+param_shape,maxshape=(None,)+param_shape,dtype=np.float32)
            for i in tqdm(range(int(niter)),total=int(niter)):
                for auxiliar in range(len(range(0, X_train.shape[0] - batch_size + 1, batch_size))):
                    X_batch, y_batch = sghmc_multicore.sample.queue.get()
                    q,p,a=self.step(X_batch,y_batch,q,p,rng)
                    logp_samples[i] = self.logp(X_batch,y_batch,q,self.hyper)
                    for var in self.start.keys():
                        param_shape=self.start[var].shape
                        posterior[var].resize((posterior[var].shape[0]+1,)+param_shape)
                        posterior[var][-1,:]=q[var]
                    backend_samples.flush()
            backend_samples.close()
            return 1, logp_samples
        else:
            posterior={var:[] for var in self.start.keys()}
            for i in tqdm(range(int(niter)),total=int(niter)):
                for auxiliar in range(len(range(0, X_train.shape[0] - batch_size + 1, batch_size))):
                    X_batch, y_batch = sghmc_multicore.sample.queue.get()
                    q,p,a=self.step(X_batch,y_batch,q,p,rng)
                    accepted.append(a)
                    logp_samples[i] = self.logp(X_batch,y_batch,q,self.hyper)
                    for var in self.start.keys():
                        posterior[var].append(q[var].reshape(-1))
                if self._verbose:
                    print('acceptance rate : {0:.4f}'.format(self.acceptance_rate(accepted)))        

            for var in self.start.keys():
                posterior[var]=np.array(posterior[var])
            return posterior,logp_samples

    def iterate_minibatches(self, X_train,y_train,queue, batch_size, total):
        for i in range(int(total)):
            #assert X_train.shape[0] == y_train.shape[0]
            for start_idx in range(0, X_train.shape[0] - batch_size + 1, batch_size):
                excerpt = slice(start_idx, start_idx + batch_size)
                queue.put((X_train[excerpt], y_train[excerpt]))

    def sample_init(self, _queue):
        sghmc_multicore.sample.queue = _queue

    def multicore_sample(self,X_train,y_train,niter=1e4,burnin=1e3,batch_size=20,backend=None,ncores=cpu_count()):
        if backend:
            multi_backend = [backend+"_%i.h5" %i for i in range(ncores)]
        else:
            multi_backend = [backend]*ncores    
        rng = [np.random.RandomState(i) for i in range(ncores)]
        queue = Queue(maxsize=ncores)      
        l = Process(target=sghmc_multicore.iterate_minibatches, args=(self, X_train,y_train,queue, batch_size, (int(niter/ncores)*ncores + int(burnin/ncores)*ncores)))
        p = Pool(None, sghmc_multicore.sample_init, [self, queue])
        l.start()
        results=p.map(unwrap_self_sghmc, zip([self]*ncores,[X_train]*ncores,[y_train]*ncores, [int(niter/ncores)]*ncores,[int(burnin/ncores)]*ncores,[batch_size]*ncores, multi_backend,rng))
        l.join()
        if not backend:
            posterior={var:np.concatenate([r[0][var] for r in results],axis=0) for var in self.start.keys()}
            logp_samples=np.concatenate([r[1] for r in results],axis=0)
            return posterior,logp_samples
        else:
            logp_samples=np.concatenate([r[1] for r in results],axis=0)
            return multi_backend,logp_samples
        