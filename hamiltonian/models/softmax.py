import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp

from mxnet.gluon.model_zoo.vision import resnet,vgg 
               

class softmax():
    
    def __init__(self,_hyper,in_units,out_units,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(out_units,in_units=in_units[0]*in_units[1]))#capa de salida
        par=self.reset(net)
        return net,par

    def reset(self,net):
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx, force_reinit=True)
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            gluon_par.grad_req='null'
        return par

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
                X=v.as_in_context(self.ctx)
                #X=nd.array(v,ctx=self.ctx)
        for name,gluon_par in self.net.collect_params().items():
            if name in par.keys():
                gluon_par.set_data(par[name])
        y_linear = self.net.forward(X)
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

    def loss(self,par,stds,**args):
        log_like=self.negative_log_likelihood(par,**args)
        log_prior=self.negative_log_prior(par,stds,**args)
        return log_like+log_prior

    def negative_log_prior(self, par,stds,**args):
        log_prior=nd.zeros(shape=1,ctx=self.ctx)
        prior=mxp.HalfNormal(scale=1.0)
        for var in par.keys():
            means=nd.zeros(par[var].shape,ctx=self.ctx)
            param_prior=mxp.normal.Normal(loc=means,scale=stds[var])
            log_prior=log_prior-nd.mean(param_prior.log_prob(nd.array(par[var])).as_nd_ndarray())-nd.mean(prior.log_prob(stds[var]).as_nd_ndarray())
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
        par=self.reset(net)
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
        net.add(gluon.nn.Dense(out_units))#capa de salida
        if self.pre_trained:
            net[1].initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
            data = nd.ones((1,in_units[0],in_units[1],in_units[2]))
            net(data.as_in_context(self.ctx))
            par=self.reset(net[1])
        else:
            net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
            data = nd.ones((1,in_units[0],in_units[1],in_units[2]))
            net(data.as_in_context(self.ctx))
            par=self.reset(net)
        return net,par

class hierarchical_resnet(resnet_softmax):

    
    def negative_log_prior(self, par,**args):
        log_prior=nd.zeros(shape=1,ctx=self.ctx)
        prior=mxp.Gamma(shape=1.0,scale=1.0)
        for var in par.keys():
            means=nd.zeros(par[var].shape,ctx=self.ctx)
            sigmas=1./prior.sample(par[var].shape).copyto(self.ctx)
            param_prior=mxp.normal.Normal(loc=means,scale=sigmas)
            log_prior=log_prior-nd.mean(param_prior.log_prob(par[var]).as_nd_ndarray())-nd.mean(prior.log_prob(sigmas).as_nd_ndarray())
        return log_prior
        
class lenet(softmax):
    
    def __init__(self,_hyper,in_units=(1,28,28),out_units=10,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(
            gluon.nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='softrelu'),
            gluon.nn.AvgPool2D(pool_size=2, strides=2),
            gluon.nn.Conv2D(channels=16, kernel_size=5, activation='softrelu'),
            gluon.nn.AvgPool2D(pool_size=2, strides=2),
            gluon.nn.Dense(120, activation='sigmoid'), 
            gluon.nn.Dense(84, activation='sigmoid'))
        net.add(gluon.nn.Dense(out_units))#capa de salida
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        data = nd.ones((1,in_units[0],in_units[1],in_units[2]))
        net(data.as_in_context(self.ctx))
        par=self.reset(net)
        return net,par
    
class hierarchical_lenet(lenet):

    def loss(self,par,stds,**args):
        log_like=self.negative_log_likelihood(par,**args)
        log_prior=self.negative_log_prior(par,stds,**args)
        return log_like+log_prior

    def negative_log_prior(self, par,stds,**args):
        log_prior=nd.zeros(shape=1,ctx=self.ctx)
        prior=mxp.HalfNormal(scale=1.0)
        for var in par.keys():
            means=nd.zeros(par[var].shape,ctx=self.ctx)
            param_prior=mxp.normal.Normal(loc=means,scale=stds[var])
            log_prior=log_prior-nd.mean(param_prior.log_prob(nd.array(par[var]).as_in_context(self.ctx)).as_nd_ndarray())-nd.mean(prior.log_prob(stds[var]).as_nd_ndarray())
        return log_prior

class vgg_softmax(softmax):
    
    def __init__(self,_hyper,in_units=(3,256,256),out_units=38,n_layers=16,pre_trained=False,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.version=2
        #n_layers = 11, 13, 16, 19.
        self.pre_trained=pre_trained
        self.net,self.par  = self._init_net(in_units,out_units,n_layers)
        
    def _init_net(self,in_units,out_units,n_layers):
        data = nd.ones((1,in_units[0],in_units[1],in_units[2]))
        net = gluon.nn.Sequential()
        model=vgg.get_vgg(n_layers, pretrained=self.pre_trained, ctx=self.ctx)
        net.add(model.features[:-4])
        #features_shape=net(data.as_in_context(self.ctx)).shape[1:]
        net.add(gluon.nn.Dense(out_units))
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        net(data.as_in_context(self.ctx))
        if self.pre_trained:
            net[1].initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
            data = nd.ones((1,in_units[0],in_units[1],in_units[2]))
            net(data.as_in_context(self.ctx))
            par=self.reset(net[1])
        else:
            net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ct5x)
            data = nd.ones((1,in_units[0],in_units[1],in_units[2]))
            net(data.as_in_context(self.ctx))
            par=self.reset(net)
        return net,par

class embeddings_softmax(softmax):
    
    def __init__(self,_hyper,in_units,out_units,n_layers,n_hidden,vocab_size,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,out_units,n_layers,n_hidden,vocab_size)
        
    def _init_net(self,in_units,out_units,n_layers,n_hidden,vocab_size):
        embedding_dim = 100
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Embedding(input_dim=vocab_size, output_dim=in_units))#capa de entrada
        net.add(gluon.nn.GlobalMaxPool1D())
        net.add(gluon.nn.Dense(n_hidden,in_units=in_units,activation='relu'))
        for i in range(1,n_layers):
            net.add(gluon.nn.Dense(n_hidden,in_units=n_hidden,activation='sigmoid'))
        net.add(gluon.nn.Dense(out_units,in_units=n_hidden))
        #net.add(gluon.nn.Dense(32,in_units=in_units,activation='relu'))
        #net.add(gluon.nn.Dense(out_units,in_units=4, activation='sigmoid'))
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            gluon_par.grad_req='null'
        return net,par
