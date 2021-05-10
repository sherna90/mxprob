import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import scipy as sp
from hamiltonian.utils import *
from numpy.linalg import inv,norm
from copy import deepcopy
from hamiltonian.inference.base import base
from tqdm import tqdm, trange

class sgld(base):

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

    def step(self,batch_size,momentum,par):
        epsilon=self.step_size
        for var in par.keys():
            momentum[var][:]=  momentum[var] - (0.5  * epsilon * par[var].grad)
            par[var][:] = par[var] + momentum[var]
        return momentum, par

    def floss(self,par, X_batch, y_batch,w_0,e):
        return self.model.negative_log_posterior(par,X_train=X_batch,y_train=y_batch)

    def draw_momentum(self,par,epsilon):
        #momentum={var:np.zeros_like(self.start[var]) for var in self.start.keys()}
        noise_scale = 2.0*epsilon
        sigma = np.sqrt(max(noise_scale, 1e-16))
        momentum={var:nd.zeros_like(0,noise_scale,size=self.start[var].shape,ctx=self.ctx) for var in par.keys()}
        return momentum


    