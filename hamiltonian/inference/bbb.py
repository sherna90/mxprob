import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base
from hamiltonian.inference.sgd import sgd
import mxnet.gluon.probability as mxp

class bbb(base):

    def softplus(self,x):
        return nd.log(1. + nd.exp(x))

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        means=deepcopy(self.start)
        means_momentum={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
        std_momentum={var:nd.zeros_like(means[var].as_nd_ndarray(),ctx=self.ctx) for var in means.keys()}
        stds={var:nd.random.normal(shape=means[var].as_nd_ndarray().shape,ctx=self.ctx) for var in means.keys()}
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
                        par[var][:]=means[var] + (stds[var] * epsilons[var]) 
                    loss = self.loss(par,means,sigmas,n_examples,batch_size,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus meansametros. por retropropagacion
                #loss es el gradiente
                means_momentum, means = self.step(batch_size,means_momentum, means)
                std_momentum, stds = self.step(batch_size,std_momentum, stds)
                j = j+1 
                cumulative_loss += nd.sum(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            #w_0 -= loss_val[i]
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val,(means,stds)

    #loss: Bayesian inference
    def loss(self,par,means,sigmas,n_data,batch_size,**args):
        for k,v in args.items():
            if k=='X_train':
                X_train=v
            elif k=='y_train':
                y_train=v
        num_batches=n_data/batch_size
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
                                            scale=self.softplus(sigmas[var]))
            posterior.update({var:variational_posterior})
        for i in range(num_samples):
            samples=[]
            labels=[]
            loglike=[]
            par=dict()
            for name,gluon_par in self.model.net.collect_params().items():
                par.update({var:posterior[var].sample().as_nd_ndarray()})
            for X_test,y_test in data_loader:
                X_test=X_test.as_in_context(self.ctx)
                y_test=y_test.as_in_context(self.ctx)
                y_pred=self.model.predict(par,X_test)
                loglike.append(y_pred.log_prob(y_test).asnumpy())
                samples.append(y_pred.sample().asnumpy())
                labels.append(y_test.asnumpy())
            total_samples.append(samples)
            total_loglike.append(loglike)
            total_labels.append(labels)
        total_samples=np.stack(total_samples)
        total_loglike=np.stack(total_loglike)
        total_labels=np.stack(total_labels)
        return total_samples,total_labels,total_loglike