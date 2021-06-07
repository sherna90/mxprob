import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base

class sgd(base):

    def fit_gluon(self,epochs=1,batch_size=1,**args):
        X=args['X_train']
        y=args['y_train']
        n_examples=X.shape[0]
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        par=self.model.par
        for var in par.keys():
            par[var].attach_grad()
        sgd = mx.optimizer.Optimizer.create_optimizer('sgd',
            learning_rate=self.step_size,rescale_grad=1./batch_size)
        states=list()
        indices=list()
        for i,var in enumerate(par.keys()):
            states.append(sgd.create_state(i,par[var]))
            indices.append(i)
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                with autograd.record():
                    loss = self.loss(par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                sgd.update(indices,[par[var] for var in par.keys()],[par[var].grad for var in par.keys()],states)
                cumulative_loss += nd.sum(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val

    def fit(self,epochs=1,batch_size=1,**args):
        X=args['X_train']
        y=args['y_train']
        n_examples=X.shape[0]
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        par=self.model.par
        for var in self.model.par.keys():
            par[var].attach_grad()
        momentum={var:nd.zeros_like(par[var]) for var in par.keys()}
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                with autograd.record():
                    loss = self.loss(par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                momentum,par=self.step(batch_size,momentum,par)
                cumulative_loss += nd.sum(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val

    
    def step(self,batch_size,momentum,par):
        for var in par.keys():
            momentum[var][:] = self.gamma * momentum[var] + self.step_size * par[var].grad /batch_size #calcula para parametros peso y bias
            par[var][:]=par[var]-momentum[var]
        return momentum, par