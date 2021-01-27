
import numpy as np
import scipy as sp
import os
from utils import *
from numpy.linalg import inv
from copy import deepcopy
from tqdm import tqdm, trange
from multiprocessing import Pool

def unwrap_self_mcmc(arg, **kwarg):
    return MH.sample(*arg, **kwarg)

class MH:
    def __init__(self, X,y,logp,start,hyper,scale=1.0,update_sequential=True,verbose=True):
        self.start = start
        self.hyper = hyper
        self.logp = logp
        self._update_sequential=update_sequential
        self._scale=scale
        self._log_factor=1.5
        self._verbose=verbose
        self.X=X
        self.y=y
        

    def step(self,q,rng):
        q_new=self.random_walk(q,rng)
        accepted=self.accept(q, q_new)
        if accepted:
            q = deepcopy(q_new)
        return q,accepted

    def acceptance_rate(self,accepted,samples):
        return np.sum(1.0*np.array(accepted))/len(samples)


    def accept(self,current_q, proposal_q):
        accept=False
        E_new = self.energy(proposal_q)
        E = self.energy(current_q)
        A = np.exp(E - E_new)
        g = np.random.rand()
        if np.isfinite(A) and (g < A):
            accept=True
        return accept


    def energy(self, q):
        return -self.logp(self.X,self.y,q,self.hyper)


    def random_walk(self,q,rng):
        q_new = {var:q[var].flatten() for var in self.start.keys()}
        for var in self.start.keys():
            factor=np.exp(rng.uniform(-self._log_factor,self._log_factor))
            dim=(np.array(self.start[var])).size
            if self._update_sequential:
                q_new[var]+=factor*self._scale*rng.normal(0.0,1.0,dim)
            else:
                ind=rng.randint(dim)
                q_new[var][ind]+=factor*self._scale*rng.normal(0.0,1.0)
            q_new[var]=q_new[var].reshape(self.start[var].shape)
        return q_new


    def sample(self,niter=1e3,burnin=100,rng=None):
        samples=[]
        burnin_samples=[]
        accepted=[]
        if rng==None:
            rng = np.random.RandomState(0)
        q=self.start
        for i in tqdm(range(int(burnin))):
            q,a=self.step(q,rng)
            burnin_samples.append(q)
            accepted.append(a)
            if i==burnin-1:
                acc_rate=self.acceptance_rate(accepted,burnin_samples)
                print('burnin acceptance rate : {0:.4f}'.format(acc_rate) )
                del accepted[:]
                del burnin_samples[:]
                self._scale=self.tune(acc_rate)
        for i in tqdm(range(int(niter))):
            q,a=self.step(q,rng)
            samples.append(q)
            accepted.append(a)
            if self._verbose and (i%(niter/10)==0):
                print('acceptance rate : {0:.4f}'.format(self.acceptance_rate(accepted,samples)) )
        posterior={var:[] for var in self.start.keys()}
        for s in samples:
            for var in self.start.keys():
                posterior[var].append(s[var].reshape(-1))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior
    
    def multicore_sample(self,niter=1e4,burnin=1e3,ncores=2):
        pool = Pool(processes=ncores)
        rng = [np.random.RandomState(i) for i in range(ncores)]
        results=pool.map(unwrap_self_mcmc, zip([self]*ncores, [int(niter/ncores)]*ncores,[burnin]*ncores,rng))
        posterior={var:np.concatenate([r[var] for r in results],axis=0) for var in self.start.keys()}
        return posterior
    
    def tune(self,acc_rate):
        new_scale=self._scale
        if acc_rate < 0.001:
            print('reduce by 90 percent')
            new_scale *= 0.1
        elif acc_rate < 0.05:
            print('reduce by 50 percent')
            new_scale *= 0.5
        elif acc_rate < 0.2:
            print('reduce by ten percent')
            # reduce by ten percent
            new_scale *= 0.9
        elif acc_rate > 0.95:
            print('increase by factor of ten')
            # increase by factor of ten
            new_scale *= 10.0
        elif acc_rate > 0.75:
            print('increase by double')
            new_scale *= 2.0
        elif acc_rate > 0.5:
            print('increase by ten percent')
            # increase by ten percent
            new_scale *= 1.1
        return new_scale           
        
            
