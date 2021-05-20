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
        par=deepcopy(self.start)
        for var in par.keys():
            par[var].attach_grad()
        sgld = mx.optimizer.Optimizer.create_optimizer('sgld',
            learning_rate=self.step_size,rescale_grad=batch_size)
        states=list()
        indices=list()
        for i,var in enumerate(par.keys()):
            states.append(sgld.create_state(i,par[var]))
            indices.append(i)
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                with autograd.record():
                    loss = self.loss(par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                sgld.update(indices,[par[var] for var in par.keys()],[par[var].grad for var in par.keys()],states)
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
        par=deepcopy(self.start)
        for var in par.keys():
            par[var].attach_grad()
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                with autograd.record():
                    loss = self.loss(par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                momentum=self.draw_momentum(par,self.step_size)
                _,par=self.step(batch_size,momentum,par)
                cumulative_loss += nd.sum(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val

    def step2(self,state,momentum,rng,**args):
        q = state.copy()
        epsilon=self.step_size
        noise_scale = 2.0*epsilon
        sigma = np.sqrt(max(noise_scale, 1e-16))     
        p = self.draw_momentum(rng,sigma)
        q_new = deepcopy(q)
        p_new = deepcopy(p)
        grad_q=self.model.grad(q,**args)
        for var in q_new.keys():
            dim=(np.array(self.start[var])).size
            nu=rng.normal(0,1,size=q_new[var].shape)
            p_new[var] = nu*p_new[var] + (1.0/n_batch)*epsilon * grad_q[var]/norm(grad_q[var])
            q_new[var]+=p_new[var]
        return q_new,p_new

    def step(self,batch_size,momentum,par):
        epsilon=self.step_size
        for var in par.keys():
            momentum[var][:]=  momentum[var] - (0.5  * epsilon * par[var].grad/batch_size)
            par[var][:] = par[var] - momentum[var]
        return momentum, par

    def draw_momentum(self,par,epsilon):
        sigma = np.sqrt(max(epsilon, 1e-16))
        momentum={var:random.normal(0,np.sqrt(epsilon),
            shape=par[var].shape,
            ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        return momentum
