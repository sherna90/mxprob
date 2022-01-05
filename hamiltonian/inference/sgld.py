from mxnet.ndarray.contrib import isnan
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon,random
from mxnet.ndarray import clip
import mxnet.gluon.probability as mxp
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base
from hamiltonian.inference.sgd import sgd
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
        params=self.model.net.collect_params()
        momentum={var:nd.numpy.zeros_like(params[var].data()) for var in params.keys()} #single_chain={var:list() for var in par.keys()}
        epsilon=self.step_size
        j=0
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    loss = self.model.loss(params,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                epsilon=self.step_size*((30 + j) ** (-0.55))
                momentum,par=self.step(epsilon,momentum,params)
                cumulative_loss += nd.numpy.mean(loss)
                j=j+1
            loss_val[i]=cumulative_loss/n_examples
            if dataset:
                for var in par.keys():
                    dataset[var][chain,i,:]=par[var].asnumpy()
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        #posterior_samples_single_chain={var:np.asarray(single_chain[var]) for var in single_chain}
        return params,loss_val
    
    def step(self,epsilon,momentum,params):
        normal=self.draw_momentum(params,epsilon)
        for var,par in zip(params,params.values()):
            if par.grad_req=='write':
                grad=par.grad()
                #momentum[var] = self.gamma*momentum[var] + (1. - self.gamma) * nd.np.square(grad)
                #par=par-self.step_size*grad/ nd.np.sqrt(momentum[var] + 1e-8)
                par.data()[:]=par.data()-self.step_size*grad+normal[var]
        return momentum, par

    def draw_momentum(self,params,epsilon):
        momentum={var:np.sqrt(epsilon)*random.normal(0,1,
            shape=params[var].shape,
            ctx=self.ctx,
            dtype=params[var].dtype) for var in params.keys()}
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
        params=self.model.net.collect_params()
        dset=[posterior_samples.create_dataset(var,shape=(chains,epochs)+params[var].shape,dtype=params[var].dtype) for var in params.keys()]
        for i in range(chains):
            #self.start=self.model.reset(self.model.net)
            _,loss=self.fit(epochs=epochs,batch_size=batch_size,
                chain=i,dataset=posterior_samples,verbose=verbose,**args)
            loss_values.append(loss)
        posterior_samples.attrs['num_chains']=chains
        posterior_samples.attrs['num_samples']=epochs 
        posterior_samples.attrs['loss']=np.stack(loss_values) 
        posterior_samples.flush()
        posterior_samples.close()  
        return posterior_samples,loss_values

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
                    par.update({var:mx.np.array(posterior_samples[var][i,j,:],ctx=self.ctx)})
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


class hierarchical_sgld(sgld):
    
    def softplus(self,x):
        return nd.log(1. + nd.exp(x))
        #return nd.exp(x)

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
        par=self.start
        scale_prior=mxp.Normal(loc=0.,scale=1.0)
        means_prior=mxp.Normal(loc=0.,scale=1.0)
        eps_prior=mxp.Normal(loc=0.,scale=1.0)
        
        mean_momentum={var:par[var].as_nd_ndarray().zeros_like(ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        std_momentum={var:par[var].as_nd_ndarray().zeros_like(ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        eps_momentum={var:par[var].as_nd_ndarray().zeros_like(ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        stds={var:scale_prior.sample(par[var].shape).copyto(self.ctx) for var in par.keys()}
        means={var:means_prior.sample(par[var].shape).copyto(self.ctx) for var in par.keys()}
        for var in par.keys():
            means[var].attach_grad()
            stds[var].attach_grad()
            par[var].attach_grad()
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            j=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                sigmas={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
                with autograd.record():
                    epsilons={var:eps_prior.sample(par[var].shape).copyto(self.ctx).as_nd_ndarray() for var in par.keys()}
                    for var in means.keys():
                        sigmas[var][:]=self.softplus(stds[var].as_nd_ndarray())
                        par[var][:]=means[var][:].as_np_ndarray()+epsilons[var][:].as_np_ndarray()*sigmas[var][:].as_np_ndarray()
                    loss=self.non_centered_hierarchical_loss(par,means,epsilons,sigmas,X_train=X_batch,y_train=y_batch)
                loss.backward()
                lr_decay=self.step_size*((30 + j) ** (-0.55))
                mean_momentum, means = self.step(lr_decay,mean_momentum, means)
                std_momentum, stds = self.step(lr_decay,std_momentum, stds)
                #eps_momentum, par = self.step(lr_decay,eps_momentum, par)
                j = j+1 
                cumulative_loss += nd.mean(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            if dataset:
                for var in par.keys():
                    dataset[var][chain,i,:]=par[var].asnumpy()
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val