import numpy as np
import scipy as sp
import os
from hamiltonian.utils import *
from numpy.linalg import inv
from copy import deepcopy
from multiprocessing import Pool,cpu_count
from tqdm import tqdm, trange
import h5py 
import os 
from scipy.optimize import check_grad
import math 
import time
import os
from hamiltonian.inference.cpu.hmc import hmc

def unwrap_self_hmc(arg, **kwarg):
    return hmc_multicore.sample(*arg, **kwarg)

class hmc_multicore(hmc):

    def sample_multicore(self,niter=1e4,burnin=1e3,ncores=cpu_count(),**args):
        
        pool = Pool(processes=ncores)
        
        kwarg={'X_train':args['X_train'],'y_train':args['y_train']}

        results=pool.map(unwrap_self_hmc, zip(
                    [int(niter/ncores)]*ncores,
                    [int(burnin)]*ncores,
                    [np.random.RandomState(i) for i in range(ncores)],
                    [kwarg]*ncores))
        
        posterior={var:np.concatenate([r[0][var] for r in results],axis=0) for var in self.start.keys()}
        sample_positions=[r[1] for r in results]
        sample_momentums=[r[2] for r in results]
        logp_samples=np.concatenate([r[3] for r in results],axis=0)
        return posterior,sample_positions,sample_momentums,logp_samples
        