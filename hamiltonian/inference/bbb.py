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
        X=args['X_train']
        y=args['y_train']
        n_data=X.shape[0]
        n_examples=X.shape[0]
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        means=deepcopy(self.start)
        means_momentum={var:nd.zeros_like(means[var],ctx=self.ctx) for var in means.keys()}
        std_momentum={var:nd.zeros_like(means[var],ctx=self.ctx) for var in means.keys()}
        stds={var:nd.random.normal(shape=means[var].shape,ctx=self.ctx) for var in means.keys()}
        for var in means.keys():
            means[var].attach_grad()
            stds[var].attach_grad()
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                par={var:nd.zeros_like(means[var],ctx=self.ctx) for var in means.keys()}
                sigmas={var:nd.zeros_like(means[var],ctx=self.ctx) for var in means.keys()}
                with autograd.record():
                    epsilons={var:nd.random.normal(shape=means[var].shape, loc=0., scale=1.0,ctx=self.ctx) for var in means.keys()}
                    for var in means.keys():
                        sigmas[var][:]=self.softplus(stds[var])
                        par[var][:]=means[var] + (stds[var] * epsilons[var]) 
                    loss = self.loss(par,means,sigmas,n_data,batch_size,X_train=X_batch,y_train=y_batch)
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
        nll=self.model.loss(par,X_train=X_train,y_train=y_train)
        log_var_posterior=list()
        for var in par.keys():
            variational_posterior=mxp.normal.Normal(loc=means[var],scale=self.softplus(sigmas[var]))
            log_var_posterior.append(nd.sum(variational_posterior.log_prob(par[var]).as_nd_ndarray()))
        return nll+ 1.0 / num_batches * sum(log_var_posterior)
    
    def step(self,batch_size,momentum,par):
        momentum, par = sgd.step(self, batch_size,momentum,par)
        return momentum, par
