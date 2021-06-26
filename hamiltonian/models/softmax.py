import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp

from mxnet.gluon.model_zoo.vision import resnet 
               

class softmax():
    
    def __init__(self,_hyper,in_units,out_units,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(out_units,in_units=in_units[0]*in_units[1]))#capa de salida
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            gluon_par.grad_req='null'
        return net,par

        
    def softmax(self, y_linear):
        exp = nd.exp(y_linear-nd.max(y_linear, axis=1).reshape((-1,1)))
        norms = nd.sum(exp, axis=1).reshape((-1,1))
        return exp / norms

    def predict(self, par,X,prob=False):
        y_hat=self.forward(par,X_train=X)   
        return y_hat	

    def forward(self,par, **args):
        for k,v in args.items():
            if k=='X_train':
                X=nd.array(v,ctx=self.ctx)
        for name,gluon_par in self.net.collect_params().items():
            gluon_par.set_data(par[name])
        y_linear = self.net.forward(X)
        #y_linear = nd.dot(X, nd.transpose(par['0.weight'])) + par['0.bias']
        yhat = self.softmax(y_linear)
        cat=mxp.Categorical(1,prob=yhat)
        return cat
     
    def negative_log_prior(self, par,**args):
        log_prior=nd.zeros(shape=1,ctx=self.ctx)
        for var in par.keys():
            means=nd.zeros(par[var].shape,ctx=self.ctx)
            sigmas=nd.ones(par[var].shape,ctx=self.ctx)*np.sqrt(self.hyper['alpha'])
            param_prior=mxp.normal.Normal(loc=means,scale=sigmas)
            log_prior=log_prior-nd.mean(param_prior.log_prob(par[var]).as_nd_ndarray())
        return log_prior
    
    def negative_log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v
            elif k=='y_train':
                y=v
        y_hat = self.forward(par,X_train=X)
        return -nd.sum(y_hat.log_prob(y).as_nd_ndarray())
        
    def loss(self,par,**args):
        log_like=self.negative_log_likelihood(par,**args)
        log_prior=self.negative_log_prior(par,**args)
        return log_like+log_prior


class hierarchical_softmax(softmax):
    
    def negative_log_prior(self, par,**args):
        log_prior=nd.zeros(shape=1,ctx=self.ctx)
        prior=mxp.Gamma(shape=1.0,scale=1.0)
        for var in par.keys():
            means=nd.zeros(par[var].shape,ctx=self.ctx)
            sigmas=1./prior.sample(par[var].shape).copyto(self.ctx)
            param_prior=mxp.normal.Normal(loc=means,scale=sigmas)
            log_prior=log_prior-nd.mean(param_prior.log_prob(par[var]).as_nd_ndarray())-nd.mean(prior.log_prob(sigmas).as_nd_ndarray())
        return log_prior

class mlp_softmax(softmax):
    
    def __init__(self,_hyper,in_units,out_units,n_layers,n_hidden,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,out_units,n_layers,n_hidden)
        
    def _init_net(self,in_units,out_units,n_layers,n_hidden):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Dense(n_hidden,in_units=in_units))#capa de entrada
        for i in range(1,n_layers):
            net.add(gluon.nn.Dense(n_hidden,in_units=n_hidden,activation='relu'))#capa de entrada
        net.add(gluon.nn.Dense(out_units,in_units=n_hidden))#capa de entrada
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            gluon_par.grad_req='null'
        return net,par

class resnet_softmax(softmax):
    
    def __init__(self,_hyper,in_units=(32,32),out_units=10,n_layers=18,pre_trained=False,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.version=2
        #n_layers = 18, 34, 50, 101, 152.
        self.pre_trained=pre_trained
        self.net,self.par  = self._init_net(in_units,out_units,n_layers)
        
    def _init_net(self,in_units,out_units,n_layers):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        model=resnet.get_resnet(self.version,n_layers,pretrained=self.pre_trained,ctx=self.ctx)
        net.add(model.features[:-1])
        net.add(gluon.nn.Dense(out_units,in_units=512))#capa de salida
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        data = nd.ones((1,3,in_units[0],in_units[1]))
        net(data.as_in_context(self.ctx))
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            gluon_par.grad_req='null'
        return net,par

class lenet(softmax):
    
    def __init__(self,_hyper,in_units=(28,28),out_units=10,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(
            gluon.nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='sigmoid'),
            #gluon.nn.AvgPool2D(pool_size=2, strides=2),
            gluon.nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),
            #gluon.nn.AvgPool2D(pool_size=2, strides=2),
            # `Dense` will transform an input of the shape (batch size, number of
            # channels, height, width) into an input of the shape (batch size,
            # number of channels * height * width) automatically by default
            gluon.nn.Dense(120, activation='sigmoid'), 
            gluon.nn.Dense(84, activation='sigmoid'))
        net.add(gluon.nn.Dense(out_units))#capa de salida
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        data = nd.ones((1,1,in_units[0],in_units[1]))
        net(data.as_in_context(self.ctx))
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            gluon_par.grad_req='null'
        return net,par