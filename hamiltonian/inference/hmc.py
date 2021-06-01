import numpy as np
from hamiltonian.utils import *
import mxnet as mx
from mxnet import nd, autograd, gluon,random
import mxnet.gluon.probability as mxp
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base
from hamiltonian.inference.find_reasonable_epsilon import DualAveragingStepSize

class hmc(base):

    def fit(self,epochs=100,burn_in=30,path_length=1.0,verbose=False,rng=None,**args):
        if rng == None:
            rng = mx.random.seed(seed_state=np.random.random_integers(0,100),ctx=self.ctx)
        q,p=self.start,self.draw_momentum(rng)
        loss=np.zeros(int(epochs))
        rate=list()
        epsilons=list()
        samples={var:[] for var in self.start.keys()}
        step_size_tuning = DualAveragingStepSize(initial_step_size=self.step_size)
        for idx in tqdm(range(int(epochs+burn_in))):
            q_new,p_new=self.step(q,p,path_length,rng,**args)
            acceptprob = self.accept(q, q_new, p, p_new,**args)
            if acceptprob and (np.random.rand() < acceptprob): 
                q = q_new.copy()
                p = p_new.copy()
            # Tuning routine
            if idx < burn_in - 1:
                self.step_size, _ = step_size_tuning.tune(acceptprob)
            elif idx == burn_in - 1:
                _, self.step_size = step_size_tuning.tune(acceptprob)   
            if idx > burn_in:
                rate.append(acceptprob)
                for var in self.start.keys():
                    samples[var].append(q[var])
        return samples,rate

    def step(self,state,momentum,path_length, rng,**args):
        q = state.copy()
        p = momentum.copy()
        q_new = deepcopy(q)
        p_new = self.draw_momentum(rng)
        epsilon=float(self.step_size)
        clip  = lambda val :  max(min(300,val),2)
        L=clip(np.ceil(path_length/epsilon))
        for var in q_new.keys():
            q_new[var].attach_grad()
        with autograd.record():
            loss = self.model.loss(q_new,**args)
        loss.backward()
        for var in q_new.keys():
            p_new[var]-= (0.5*epsilon)*q_new[var].grad  
        # leapfrog step 
        for _ in range(int(L-1)):
            for var in self.start.keys():
                q_new[var]+= epsilon*p_new[var]
            with autograd.record():
                loss = self.model.loss(q_new,**args)
            loss.backward()
            for var in self.start.keys():
                p_new[var]-= (0.5*epsilon)*q_new[var].grad
        for var in self.start.keys():
            q_new[var]+= epsilon*p_new[var]
        with autograd.record():
            loss = self.model.loss(q_new,**args)
        for var in self.start.keys():
            p_new[var]-= epsilon*q_new[var].grad
            p_new[var]=-p_new[var]
        return q_new,p_new


    def accept(self,current_q, proposal_q, current_p, proposal_p,**args):
        E_proposal = (self.model.loss(proposal_q,**args)+self.potential_energy(proposal_p))
        E_current = (self.model.loss(current_q,**args)+self.potential_energy(current_p))
        A = min(1.,nd.exp(E_proposal-E_current).asnumpy())
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


        

