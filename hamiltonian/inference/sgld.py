from mxnet.ndarray.contrib import isnan
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon,random
from mxnet.ndarray import clip
import mxnet.gluon.probability as mxp
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
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                epsilon=self.step_size*((30 + j) ** (-0.55))
                momentum,par=self.step(n_examples,batch_size,momentum,epsilon,self.model.par)
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


    def step(self,n_data,batch_size,momentum,epsilon,par):
        normal=self.draw_momentum(par,epsilon)
        for var in par.keys():
            grad = par[var].grad.as_nd_ndarray()
            momentum[var][:] = self.gamma*momentum[var] + (1. - self.gamma) * nd.square(grad)
            par[var][:]=par[var]-self.step_size*grad/ nd.sqrt(momentum[var].as_nd_ndarray() + 1e-8)+normal[var].as_nd_ndarray()
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
            #self.model.par=self.model.reset(self.model.net)
            for var in self.model.par.keys():
                self.model.par[var]=nd.array(self.start[var],ctx=self.ctx)
            _,loss=self.fit(epochs=epochs,batch_size=batch_size,
                chain=i,dataset=posterior_samples,verbose=verbose,**args)
            loss_values.append(loss)
        posterior_samples.attrs['num_chains']=chains
        posterior_samples.attrs['num_samples']=epochs 
        posterior_samples.attrs['loss']=np.stack(loss_values) 
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
                    par.update({var:nd.array(posterior_samples[var][i,j,:],ctx=self.ctx)})
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
        means=self.model.par
        scale_prior=mxp.HalfNormal(scale=1.0)
        eps_prior=mxp.Normal(loc=0.,scale=1.0)
        stds={var:scale_prior.sample(means[var].shape).copyto(self.ctx).as_nd_ndarray() for var in means.keys()}
        #epsilons={var:eps_prior.sample(means[var].shape).copyto(self.ctx).as_nd_ndarray() for var in means.keys()}
        for var in self.model.par.keys():
            means[var].attach_grad()
            stds[var].attach_grad()
            #epsilons[var].attach_grad()
        j=0
        mean_momentum={var:means[var].as_nd_ndarray().zeros_like(ctx=self.ctx,
            dtype=means[var].dtype) for var in means.keys()}
        std_momentum={var:nd.zeros_like(stds[var].as_nd_ndarray(),ctx=self.ctx) for var in stds.keys()}
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                par={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
                sigmas={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
                epsilons={var:eps_prior.sample(means[var].shape).copyto(self.ctx).as_nd_ndarray() for var in means.keys()}
                with autograd.record():
                    for var in means.keys():
                        sigmas[var]=self.softplus(stds[var])
                        par[var][:]=means[var].as_nd_ndarray() + (sigmas[var] * epsilons[var])
                    loss = self.hierarchical_loss(par,means,epsilons,sigmas,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                lr_decay=self.step_size*((30 + j) ** (-0.55))
                mean_momentum,means=self.step(n_examples,batch_size,mean_momentum,lr_decay,means)
                std_momentum, stds = self.step(n_examples,batch_size,std_momentum,lr_decay, stds)
                #mean_momentum, epsilons = self.step(n_examples,batch_size,mean_momentum,lr_decay, epsilons)
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