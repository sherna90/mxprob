import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base

class sgd(base):
    
    def step(self,batch_size,momentum,par):
        for var in par.keys():
            momentum[var][:] = self.gamma * momentum[var] + self.step_size * par[var].grad /batch_size #calcula para parametros peso y bias
            par[var][:]=par[var]-momentum[var]
        return momentum, par