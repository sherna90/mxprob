import warnings
warnings.filterwarnings("ignore")

import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp

class softmax():
    
    def __init__(self,_hyper,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.LOG2PI = np.log(2.0 * np.pi)
        self.par = dict()
        self.net = self.init_net()
        
        
        
    def init_net(self):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        #net.add(gluon.nn.Dense(128, activation="relu"))#primera capa
        #net.add(gluon.nn.Dense(64, activation="relu"))#segunda capa
        net.add(gluon.nn.Dense(10))#capa de salida
        #print(type(net))
        #print(type(net.collect_params()))
        net.initialize(mx.init.Xavier(magnitude=2.24), ctx=self.ctx)
        net.collect_params()
        x = nd.random.uniform(shape=(1,784))
        net.forward(x)
        for name,gluon_par in net.collect_params().items():
            self.par.update({name:gluon_par.data()})
        #print(net.collect_params())
        #print(net[0].params)
        #print(net[0].bias)
        #print(net[0].bias.data())
        #print(net[0].weight.grad())
        return net

        
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
                X=v
        #y_linear = nd.dot(X, par['weights']) + par['bias']
        X = X.as_in_context(self.ctx).reshape((-1,784))
        for name,gluon_par in self.net.collect_params().items():
            gluon_par.set_data(self.par[name])
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
        return -nd.mean(y_hat.log_prob(y).as_nd_ndarray())
        
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