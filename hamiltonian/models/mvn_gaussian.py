import warnings
warnings.filterwarnings("ignore")

import numpy as np
from numpy.linalg import *
from hamiltonian.utils import *
import hamiltonian.models.model as base_model

class mvn_gaussian:

    def __init__(self,_hyper):
        self.hyper=_hyper

    def grad(self,par,**args):
        cov=self.hyper['cov']
        mu=self.hyper['mu']
        x=par['x']
        grad={}
        grad['x']=np.dot(x-mu,inv(cov))
        return grad	
        
    def negative_log_posterior(self,par,**args):
        dim=self.hyper['mu'].shape[0]
        sigma=self.hyper['cov']
        mu=self.hyper['mu']
        x=par['x']
        log_loss=dim * np.log(2 * np.pi)
        log_loss+=np.log(np.linalg.det(sigma))
        log_loss+=np.dot(np.dot((x - mu).T, np.linalg.inv(sigma)), x - mu)
        log_loss*= 0.5
        return log_loss
