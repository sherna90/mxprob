import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy

class sgd:
    
    def __init__(self,model,start_p,step_size=0.1,ctx=mx.cpu()):
        self.start = start_p
        self.step_size = step_size
        self.model = model
        self.ctx=ctx
        

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield nd.array(X[excerpt],ctx=self.ctx), nd.array(y[excerpt],ctx=self.ctx)

    def fit(self,epochs=1,batch_size=1,gamma=0.9,**args):
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
        momentum={var:nd.zeros_like(par[var],ctx=self.ctx) for var in par.keys()}
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                with autograd.record():
                    loss = self.model.negative_log_posterior(par,X_train=X_batch,y_train=y_batch)
                loss.backward()
                for var in par.keys():
                    momentum[var][:] = gamma * momentum[var] + self.step_size * par[var].grad /batch_size
                    par[var][:]=par[var]-momentum[var]
                cumulative_loss += nd.sum(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val
