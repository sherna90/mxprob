import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base
from hamiltonian.inference.sgd import sgd
import mxnet.gluon.probability as mxp
import os
import h5py 

class bbb(base):

    def softplus(self,x):
        return nd.log(1. + nd.exp(x))

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain_name' in args:
            if os.path.exists(args['chain_name']):
                os.remove(args['chain_name'])
            variational_posterior=h5py.File(args['chain_name'],'w')
        else:
            if os.path.exists('variational_posterior.h5'):
                os.remove('variational_posterior.h5')
            variational_posterior=h5py.File('variational_posterior.h5','w')
        posterior_means=variational_posterior.create_group('means')
        posterior_stds=variational_posterior.create_group('stds')
        dset=[posterior_means.create_dataset(var,shape=self.start[var].shape,dtype=self.start[var].dtype) for var in self.start.keys()]
        dset=[posterior_stds.create_dataset(var,shape=self.start[var].shape,dtype=self.start[var].dtype) for var in self.start.keys()]
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        means=deepcopy(self.start)
        means_momentum={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
        std_momentum={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
        stds={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
        for var in means.keys():
            means[var].attach_grad()
            stds[var].attach_grad()
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            j=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                par={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
                sigmas={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
                with autograd.record():
                    epsilons={var:nd.random.normal(shape=means[var].as_nd_ndarray().shape, loc=0., scale=1.0,ctx=self.ctx) for var in means.keys()}
                    for var in means.keys():
                        sigmas[var][:]=self.softplus(stds[var])
                        par[var][:]=means[var].as_nd_ndarray() + (sigmas[var] * epsilons[var])
                    #loss = self.model.loss(par,X_train=X_batch,y_train=y_batch) 
                    loss = self.model.loss(par,means,epsilons,sigmas,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus meansametros. por retropropagacion
                #loss es el gradiente
                means_momentum, means = self.step(batch_size,means_momentum, means)
                means_momentum, stds = self.step(batch_size,means_momentum, stds)
                j = j+1 
                cumulative_loss += nd.mean(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            #w_0 -= loss_val[i]
            if verbose and (i%(epochs/epochs)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        if variational_posterior:
            for var in par.keys():
                posterior_means[var][:]=means[var].asnumpy()
                posterior_stds[var][:]=sigmas[var].asnumpy()
        variational_posterior.attrs['loss']=loss_val
        variational_posterior.flush()
        variational_posterior.close()
        return par,loss_val,(means,sigmas)

    #loss: Bayesian inference
    def loss(self,par,means,sigmas,n_data,batch_size,**args):
        for k,v in args.items():
            if k=='X_train':
                X_train=v
            elif k=='y_train':
                y_train=v
        num_batches=n_data/batch_size
        for var in self.model.par.keys():
            if type(self.model.par[var])=='mxnet.numpy.ndarray':
                par.update({var:par[var].as_np_ndarray()})
        log_likelihood_sum=self.model.negative_log_likelihood(par,X_train=X_train,y_train=y_train)
        log_prior_sum=self.model.negative_log_prior(par,X_train=X_train,y_train=y_train)
        log_var_posterior=list()
        for var in par.keys():
            variational_posterior=mxp.normal.Normal(loc=means[var],scale=self.softplus(sigmas[var]))
            log_var_posterior.append(nd.mean(variational_posterior.log_prob(par[var]).as_nd_ndarray()))
        log_var_posterior_sum=sum(log_var_posterior)
        return 1.0 / num_batches * (log_var_posterior_sum + log_prior_sum) + log_likelihood_sum
    
    def step(self,batch_size,momentum,par):
        momentum, par = sgd.step(self, batch_size,momentum,par)
        return momentum, par

    def predict(self,means,sigmas,num_samples,**args):
        data_loader,_=self._get_loader(**args)
        total_samples=[]
        total_loglike=[]
        total_labels=[]
        posterior=dict()
        for var in means.keys():
            variational_posterior=mxp.normal.Normal(loc=means[var],
                                            scale=sigmas[var])
            posterior.update({var:variational_posterior})
        for i in range(num_samples):
            samples=[]
            labels=[]
            loglike=[]
            par=dict()
            for var in means.keys():
                par.update({var:posterior[var].sample().as_nd_ndarray()})
            for X_test,y_test in data_loader:
                X_test=X_test.as_in_context(self.ctx)
                y_test=y_test.as_in_context(self.ctx)
                y_pred=self.model.predict(par,X_test)
                if isinstance(y_pred.sample(),mx.numpy.ndarray):
                    loglike.append(y_pred.log_prob(y_test.as_np_ndarray()).asnumpy())
                else:
                    loglike.append(y_pred.log_prob(y_test).asnumpy())
                samples.append(y_pred.sample().asnumpy())
                labels.append(y_test.asnumpy())
            total_samples.append(np.concatenate(samples))
            total_loglike.append(np.concatenate(loglike))
        total_labels=np.concatenate(labels)
        total_samples=np.stack(total_samples)
        total_loglike=np.stack(total_loglike)
        return total_samples,total_labels,total_loglike