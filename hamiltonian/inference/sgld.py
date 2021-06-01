import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon,random
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base

class sgld(base):

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
        for var in self.model.par.keys():
            self.model.par[var].attach_grad()
        sgld = mx.optimizer.Optimizer.create_optimizer('sgld',
            learning_rate=self.step_size,rescale_grad=1./batch_size)
        states=list()
        indices=list()
        samples={var:[] for var in self.model.par.keys()}
        for i,var in enumerate(self.model.par.keys()):
            states.append(sgld.create_state(i,self.model.par[var]))
            indices.append(i)
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                with autograd.record():
                    loss = self.loss(self.model.par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                sgld.update(indices,[self.model.par[var] for var in self.model.par.keys()],[self.model.par[var].grad for var in self.model.par.keys()],states)
                cumulative_loss += nd.sum(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
            for var in self.model.par.keys():
                samples[var].append(self.model.par[var])
        return self.model.par,loss_val,samples

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
        for var in self.model.par.keys():
            self.model.par[var].attach_grad()
        j=0
        samples={var:[] for var in self.model.par.keys()}
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                with autograd.record():
                    loss = self.loss(self.model.par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                epsilon=self.step_size*((30 + j) ** (-0.55))
                momentum=self.draw_momentum(self.model.par,epsilon)
                _,self.model.par=self.step(n_examples,batch_size,momentum,epsilon,self.model.par)
                cumulative_loss += nd.sum(loss).asscalar()
                j=j+1
            loss_val[i]=cumulative_loss/n_examples
            for var in self.model.par.keys():
                samples[var].append(self.model.par[var])
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return self.model.par,loss_val,samples

    def step(self,n_data,batch_size,momentum,epsilon,par):
        for name,par_new in par.items():
            momentum[name][:]=  momentum[name] - (0.5  * epsilon  * (n_data//batch_size) * par[name].grad)
            par[name][:]=par_new + momentum[name]
        return momentum, self.model.par

    def draw_momentum(self,par,epsilon):
        momentum={var:np.sqrt(epsilon)*random.normal(0,1,
            shape=par[var].shape,
            ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        return momentum
