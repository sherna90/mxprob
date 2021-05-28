import numpy as np
from hamiltonian.utils import *
import mxnet as mx
from mxnet import nd, autograd, gluon,random
import mxnet.gluon.probability as mxp
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base

class hmc(base):

    def fit(self,epochs=100,burn_in=30,path_length=1.0,verbose=False,rng=None,**args):
        if rng == None:
            rng = mx.random.seed(seed_state=1,ctx=self.ctx)
        q,p=self.start,self.draw_momentum(rng)
        loss=np.zeros(int(epochs))
        samples={var:[] for var in self.start.keys()}
        for i in tqdm(range(int(epochs+burn_in))):
            q_new,p_new=self.step(q,p,path_length,rng,**args)
            acceptprob = self.accept(q, q_new, p, p_new,**args)
            if acceptprob and (np.random.rand() < acceptprob): 
                q = q_new.copy()
                p = p_new.copy()
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss[i]))
            if i>burn_in:
                for var in self.start.keys():
                    samples[var].append(q[var])
        return samples,loss

    def step(self,state,momentum,path_length, rng,**args):
        q = state.copy()
        p = momentum.copy()
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        epsilon=self.step_size
        path_length=np.ceil(2*np.random.rand()*path_length/epsilon)
        for var in q_new.keys():
            q_new[var].attach_grad()
        with autograd.record():
            loss = self.model.loss(q_new,**args)
        loss.backward()
        for var in q_new.keys():
            p_new[var]-= (0.5*epsilon)*q_new[var].grad     
        # leapfrog step 
        for _ in np.arange(path_length-1):
            for var in self.start.keys():
                q_new[var]+= epsilon*p_new[var]
                with autograd.record():
                    loss = self.model.loss(q_new,**args)
                loss.backward()
                p_new[var]-= (0.5*epsilon)*q_new[var].grad
        for var in self.start.keys():
            q_new[var]+= epsilon*p_new[var]
        with autograd.record():
            loss = self.model.loss(q_new,**args)
        for var in self.start.keys():
            p_new[var]-= epsilon*q_new[var].grad
        for var in self.start.keys():
            p_new[var]=-p_new[var]
        return q_new,p_new


    def accept(self,current_q, proposal_q, current_p, proposal_p,**args):
        E_new = (self.model.loss(proposal_q,**args)+self.potential_energy(proposal_p))
        E_current = (self.model.loss(current_q,**args)+self.potential_energy(current_p))
        A = min(1.,np.exp(E_current-E_new))
        return A


    def potential_energy(self,momentum):
        K=nd.zeros(shape=1,ctx=self.ctx)
        for var in momentum.keys():
            means=nd.zeros(momentum[var].shape,ctx=self.ctx)
            sigmas=nd.ones(momentum[var].shape,ctx=self.ctx)
            param=mxp.normal.Normal(loc=means,scale=sigmas)
            K-=nd.mean(param.log_prob(momentum[var]).as_nd_ndarray())
        return K


    def draw_momentum(self,rng):
        momentum={var:random.normal(0,1,
            shape=self.start[var].shape,
            ctx=self.ctx,
            dtype=self.start[var].dtype) for var in self.start.keys()}
        return momentum

            
    def find_reasonable_epsilon(self,p_accept,**args):
        self.t += 1
        g=self.target_accept - p_accept
        self.error_sum += self.target_accept - p_accept
        #self.error_sum  = (1.0 - 1.0/(self.t + self.t0)) * self.error_sum + g/(self.t + self.t0)
        log_step = self.mu - self.prox_center - (self.t ** 0.5) / self.gamma * self.error_sum
        eta = self.t **(-self.kappa)
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        return np.exp(log_step), np.exp(self.log_averaged_step)

        

