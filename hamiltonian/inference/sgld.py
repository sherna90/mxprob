import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon,random
from mxnet.ndarray import clip
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base
import h5py
import os 

class sgld(base):

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain' in args:
            chain=args['chain']
        else:
            chain=None
        if 'dataset' in args:
            dataset=args['dataset']
        else:
            dataset=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        par=self.model.par
        for var in self.model.par.keys():
            par[var].attach_grad()
        j=0
        momentum={var:par[var].as_nd_ndarray().zeros_like(ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        #single_chain={var:list() for var in par.keys()}
        epsilon=self.step_size
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    loss = self.loss(par,X_train=X_batch,y_train=y_batch)
                loss.backward()
                epsilon=self.step_size*((30 + j) ** (-0.55))
                momentum,par=self.step(batch_size,momentum,epsilon,par)
                cumulative_loss += nd.mean(loss).asscalar()
                j=j+1
            loss_val[i]=cumulative_loss/n_examples
            if dataset:
                for var in par.keys():
                    dataset[var][chain,i,:]=par[var].asnumpy()
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        #posterior_samples_single_chain={var:np.asarray(single_chain[var]) for var in single_chain}
        return par,loss_val

    def preconditioned_step(self,batch_size,momentum,epsilon,par):
        normal=self.draw_momentum(par,epsilon)
        for var in par.keys():
            #grad = clip(par[var].grad, -1e3,1e3)
            grad = par[var].grad/batch_size
            momentum[var][:] = self.gamma*momentum[var] + (1. - self.gamma) * nd.square(grad)
            par[var][:]=par[var]-epsilon*grad/ nd.sqrt(momentum[var].as_nd_ndarray() + 1e-8)+normal[var].as_nd_ndarray()
        return momentum, par

    def step(self,batch_size,momentum,epsilon,par):
        normal=self.draw_momentum(par,epsilon)
        for var in par.keys():
            #grad = np.nan_to_num(par[var].grad).as_nd_ndarray()
            grad=par[var].grad.as_nd_ndarray()
            momentum[var][:] = self.gamma * momentum[var] + epsilon * grad #calcula para parametros peso y bias
            par[var][:]=par[var]-momentum[var]+normal[var].as_nd_ndarray()
        return momentum, par
        
    def draw_momentum(self,par,epsilon):
        momentum={var:np.sqrt(epsilon)*random.normal(0,1,
            shape=par[var].shape,
            ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        return momentum

    def sample(self,epochs=1,batch_size=1,chains=2,verbose=False,**args):
        loss_values=list()
        posterior_samples_multiple_chains=list()
        if 'chain_name' in args:
            if os.path.exists(args['chain_name']):
                os.remove(args['chain_name'])
            posterior_samples=h5py.File(args['chain_name'],'w')
        else:
            if os.path.exists('posterior_samples.h5'):
                os.remove('posterior_samples.h5')
            posterior_samples=h5py.File('posterior_samples.h5','w')
        dset=[posterior_samples.create_dataset(var,shape=(chains,epochs)+self.model.par[var].shape,dtype=self.model.par[var].dtype) for var in self.model.par.keys()]
        for i in range(chains):
            _,loss=self.fit(epochs=epochs,batch_size=batch_size,
                chain=i,dataset=posterior_samples,verbose=verbose,**args)
            loss_values.append(loss)
            self.model.par=self.model.reset(self.model.net)
            #posterior_samples_multiple_chains.append(posterior_samples_single_chain)
        #posterior_samples_multiple_chains_expanded=[ {var:np.expand_dims(sample,axis=0) for var,sample in posterior.items()} for posterior in posterior_samples_multiple_chains]
        #samples = {var:np.concatenate([posterior_samples_multiple_chains_expanded[i][var] for i in range(len(posterior_samples_multiple_chains_expanded))]) for var in self.model.par}
        #dset=[posterior_samples.create_dataset(var,data=samples[var]) for var in samples.keys()]
        posterior_samples.attrs['num_chains']=chains
        posterior_samples.attrs['num_samples']=epochs 
        posterior_samples.flush()
        posterior_samples.close()  
        return loss_values,posterior_samples

    def predict(self,posterior_samples,**args):
        data_loader,_=self._get_loader(**args)
        total_samples=list()
        num_samples=posterior_samples.attrs['num_samples']
        num_chains=posterior_samples.attrs['num_chains']
        total_loglike=list()
        for i in range(num_chains):
            for j in range(num_samples):
                par=dict()
                for var in posterior_samples.keys():
                    par.update({var:posterior_samples[var][i,j,:]})
                samples=list()
                loglike=list()
                labels=list()
                for X_test,y_test in data_loader:
                    X_test=X_test.as_in_context(self.ctx)
                    y_test=y_test.as_in_context(self.ctx)
                    labels.append(y_test.asnumpy())
                    y_pred=self.model.predict(par,X_test)
                    if isinstance(y_pred.sample(),mx.numpy.ndarray):
                        loglike.append(y_pred.log_prob(y_test.as_np_ndarray()).asnumpy())
                    else:
                        loglike.append(y_pred.log_prob(y_test).asnumpy())
                    samples.append(y_pred.sample().asnumpy())
                total_samples.append(np.concatenate(samples))
                total_loglike.append(np.concatenate(loglike))
        total_samples=np.stack(total_samples)
        total_loglike=np.stack(total_loglike)
        total_labels=np.concatenate(labels)
        return total_samples,total_labels,total_loglike

    def posterior_diagnostics(self,posterior_samples):
        num_chains=posterior_samples.attrs['num_chains']
        posterior_samples_multiple_chains=list()
        for i in range(num_chains):
            posterior_samples_single_chain={var:posterior_samples[var][:] for var in self.model.par}
            posterior_samples_multiple_chains.append(posterior_samples_single_chain)
        posterior_samples_multiple_chains_expanded=[ {var:np.expand_dims(sample,axis=0) for var,sample in posterior.items()} for posterior in posterior_samples_multiple_chains]
        return posterior_samples_multiple_chains_expanded
