import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv,norm
from copy import deepcopy
import os 
from hamiltonian.inference.cpu.sgmcmc import sgmcmc
from tqdm import tqdm, trange
import h5py 
import time

class sgld(sgmcmc):

    def step2(self,state,momentum,rng,**args):
        q = state.copy()
        epsilon=self.step_size
        noise_scale = 2.0*epsilon
        sigma = np.sqrt(max(noise_scale, 1e-16))     
        p = self.draw_momentum(rng,sigma)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        grad_q=self.model.grad(q,**args)
        for var in q_new.keys():
            dim=(np.array(self.start[var])).size
            nu=rng.normal(0,1,size=q_new[var].shape)
            p_new[var] = nu*p_new[var] + (1.0/n_batch)*epsilon * grad_q[var]/norm(grad_q[var])
            q_new[var]+=p_new[var]
        return q_new,p_new

    def step(self,state,momentum,rng,**args):
        epsilon=self.step_size
        q = deepcopy(state)
        p = self.draw_momentum(rng,epsilon)
        grad_q=self.model.grad(q,**args)
        for var in p.keys():
            p[var]+=  - 0.5  * epsilon * grad_q[var]
            q[var]+=p[var]
        return q,p

    def draw_momentum(self,rng,epsilon):
        #momentum={var:np.zeros_like(self.start[var]) for var in self.start.keys()}
        noise_scale = 2.0*epsilon
        sigma = np.sqrt(max(noise_scale, 1e-16))  
        momentum={var:rng.normal(0,noise_scale,size=self.start[var].shape) for var in self.start.keys()}
        return momentum


    