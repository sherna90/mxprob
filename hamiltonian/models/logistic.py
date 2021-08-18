import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp


class logistic():
    
    def __init__(self,_hyper,in_units,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units)
        
    def _init_net(self,in_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Dense(1,in_units=in_units))#capa de salida
        par=self.reset(net)
        return net,par

    def reset(self,net):
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx, force_reinit=True)
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            gluon_par.grad_req='null'
        return par

    def predict(self, par,X,prob=False):
        y_hat=self.forward(par,X_train=X)   
        return y_hat	

    def forward(self,par, **args):
        eps = 1e-3
        dtype=set([self.par[var].dtype for var in self.par.keys()]).pop()
        for k,v in args.items():
            if k=='X_train':
                X_train=v
        for name,gluon_par in self.net.collect_params().items():
            if name in par.keys():
                if isinstance(gluon_par.data(),mx.numpy.ndarray):
                    gluon_par.set_data(par[name].as_np_ndarray())
                else:
                    gluon_par.set_data(par[name])
        y_linear = self.net.forward(X_train)
        yhat = nd.sigmoid(y_linear)
        yhat=nd.clip(yhat,eps,1.-eps)
        cat=mxp.Binomial(n=1,prob=yhat)
        return cat
     
    def negative_log_prior(self, par,**args):
        log_prior=nd.zeros(shape=1,ctx=self.ctx)
        for var in par.keys():
            means=nd.zeros(par[var].shape,ctx=self.ctx)
            sigmas=nd.ones(par[var].shape,ctx=self.ctx)*np.sqrt(self.hyper['alpha'])
            param_prior=mxp.normal.Normal(loc=means,scale=sigmas)
            log_prior+nd.mean(param_prior.log_prob(par[var]).as_nd_ndarray())
        return -1.0*nd.sum(log_prior)
    
    def negative_log_likelihood(self,par,**args):
        dtype=set([self.par[var].dtype for var in self.par.keys()]).pop()
        for k,v in args.items():
            if k=='X_train':
                X=v
            elif k=='y_train':
                y=v
        y_hat = self.forward(par,X_train=X)
        return -nd.sum(y_hat.log_prob(y.as_np_ndarray()).as_nd_ndarray()).astype(dtype)
        
    def loss(self,par,**args):
        log_like=self.negative_log_likelihood(par,**args)
        log_prior=self.negative_log_prior(par,**args)
        return log_like+log_prior


class embeddings_logistic(logistic):
    
    def __init__(self,_hyper,in_units,n_layers,n_hidden,vocab_size,embedding_dim,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,n_layers,n_hidden,vocab_size,embedding_dim)
        
    def _init_net(self,in_units,n_layers,n_hidden,vocab_size,embedding_dim):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Embedding(input_dim=vocab_size, output_dim=embedding_dim))#capa de entrada
        net.add(gluon.nn.GlobalMaxPool1D())
        net.add(gluon.nn.Dense(n_hidden,in_units=embedding_dim,activation='relu'))
        for i in range(1,n_layers):
            net.add(gluon.nn.Dense(n_hidden,in_units=n_hidden,activation='sigmoid'))
        net.add(gluon.nn.Dense(1,in_units=n_hidden))
        #net.add(gluon.nn.Dense(32,in_units=in_units,activation='relu'))
        #net.add(gluon.nn.Dense(out_units,in_units=4, activation='sigmoid'))
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            gluon_par.grad_req='null'
        return net,par