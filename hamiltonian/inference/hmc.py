import numpy as np
from hamiltonian.utils import *
import hamiltonian.inference.cpu.find_reasonable_epsilon as find_reasonable_epsilon
from numpy.linalg import inv,norm
from copy import deepcopy
from tqdm import tqdm, trange
import h5py 
import os 
from multiprocessing import Pool,cpu_count

class hmc:
    def __init__(self,model, start_p, path_length=1.0,step_size=0.1,verbose=True):
        self.start = start_p
        self.step_size = step_size
        self.path_length = path_length
        self.model = model
        self._mass_matrix={}
        self._inv_mass_matrix={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            if dim==1:
                self._mass_matrix[var]=np.array(1.0)
                self._inv_mass_matrix[var]=np.array(1.0)
            else:
                self._mass_matrix[var]=np.ones(dim)
                self._inv_mass_matrix[var]=np.ones(dim)
        self.verbose=verbose
        # dual averaging parameters
        self.mu = np.log(10 * self.step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = 0.65
        self.gamma = 0.05
        self.t0 = 10.0
        self.t = 0
        self.kappa = 0.75
        self.error_sum = 0.0
        self.log_averaged_step = 0.0


    def step(self,state,momentum,rng,**args):
        q = state.copy()
        p = self.draw_momentum(rng)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        positions, momentums = [deepcopy(q)], [deepcopy(p)]
        epsilon=self.step_size
        path_length=np.ceil(2*np.random.rand()*self.path_length/epsilon)
        grad_q=self.model.grad(q,**args)
        # leapfrog step 
        for _ in np.arange(path_length-1):
            for var in self.start.keys():
                p_new[var]-= (0.5*epsilon)*grad_q[var]
                q_new[var]+= epsilon*p_new[var]
                grad_q=self.model.grad(q_new,**args)
                p_new[var]-= epsilon*grad_q[var]
                #positions.append(deepcopy(q_new)) 
                #momentums.append(deepcopy(p_new)) 
        # negate momentum
        for var in self.start.keys():
            p_new[var]=-p_new[var]
        acceptprob = self.accept(q, q_new, p, p_new,**args)
        if np.isfinite(acceptprob) and (np.random.rand() < acceptprob): 
            q = q_new.copy()
            p = p_new.copy()
        return q,p,positions, momentums,acceptprob


    def accept(self,current_q, proposal_q, current_p, proposal_p,**args):
        E_new = (self.model.negative_log_posterior(proposal_q,**args)+self.potential_energy(proposal_p))
        E_current = (self.model.negative_log_posterior(current_q,**args)+self.potential_energy(current_p))
        A = min(1,np.exp(E_current-E_new))
        return A


    def potential_energy(self,p):
        K=0
        for var in p.keys():
            dim=(np.array(p[var])).size
            K+=0.5*(np.sum(np.square(p[var])))
        return K


    def draw_momentum(self,rng):
        momentum={}
        for var in self.start.keys():
            dim=(np.array(self.start[var])).size
            momentum[var]=rng.normal(0,1,size=self.start[var].shape)
        return momentum


    def sample(self,niter=1e4,burnin=1e3,rng=None,**args):
        if rng == None:
            rng = np.random.RandomState()
        q,p=self.start,self.draw_momentum(rng)
        step_size_tuning = DualAveragingStepSize(self.step_size)
        for i in tqdm(range(int(burnin))):
            q,p,positions,momentums,p_accept=self.step(q,p,rng,**args)
            if self.verbose is not None and (i%(burnin/10)==0):
                ll=self.model.negative_log_posterior(q,**args)
                print('loss: {0:.4f}'.format(ll))
            #self.step_size,_=step_size_tuning.update(p_accept)
        _,avg_step_size=step_size_tuning.update(p_accept)
        print('adapted step size : ',avg_step_size)
        #if avg_step_size<0.5:
        #    self.step_size=avg_step_size
        loss=np.zeros(int(niter))
        sample_positions, sample_momentums = [], []
        posterior={var:[] for var in self.start.keys()}
        for i in tqdm(range(int(niter))):
            q,p,positions,momentums,_=self.step(q,p,rng,**args)
            sample_positions.append(positions)
            sample_momentums.append(momentums)
            loss[i]=self.model.negative_log_posterior(q,**args)
            for var in self.start.keys():
                posterior[var].append(q[var])
            if self.verbose and (i%(niter/10)==0):
                print('loss: {0:.4f}'.format(loss[i]))
        for var in self.start.keys():
            posterior[var]=np.array(posterior[var])
        return posterior,loss,sample_positions,sample_momentums

            
    def find_reasonable_epsilon(self,p_accept,**args):
        self.t += 1
        g=self.target_accept - p_accept
        self.error_sum += self.target_accept - p_accept
        #self.error_sum  = (1.0 - 1.0/(self.t + self.t0)) * self.error_sum + g/(self.t + self.t0)
        log_step = self.mu - self.prox_center - (self.t ** 0.5) / self.gamma * self.error_sum
        eta = self.t **(-self.kappa)
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        return np.exp(log_step), np.exp(self.log_averaged_step)

    def backend_mean(self, multi_backend, niter, ncores=cpu_count()):
        aux = []
        for filename in multi_backend:
            f=h5py.File(filename)
            aux.append({var:np.sum(f[var],axis=0) for var in f.keys()})
        mean = {var:((np.sum([r[var] for r in aux],axis=0).reshape(self.start[var].shape))/niter) for var in self.start.keys()}
        return mean
        

class DualAveragingStepSize:
    def __init__(
        self, initial_step_size, target_accept=0.8, gamma=0.05, t0=10.0, kappa=0.75
    ):
        self.mu = np.log(10 * initial_step_size)
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0

    def update(self, p_accept):
        """Propose a new step size.

        This method returns both a stochastic step size and a dual-averaged
        step size. While tuning, the HMC algorithm should use the stochastic
        step size and call `update` every loop. After tuning, HMC should use
        the dual-averaged step size for sampling.

        Parameters
        ----------
        p_accept: float
            The probability of the previous HMC proposal being accepted

        Returns
        -------
        float, float
            A stochastic step size, and a dual-averaged step size
        """
        self.error_sum += self.target_accept - p_accept
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        eta = self.t ** -self.kappa
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        self.t += 1
        return np.exp(log_step), np.exp(self.log_averaged_step)
