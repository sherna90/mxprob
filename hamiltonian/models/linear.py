import warnings
warnings.filterwarnings("ignore")

import mxnet as mx
from mxnet import np
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp

class linear():
    
    def __init__(self,_hyper,in_units,out_units,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.in_units=in_units
        self.out_units=out_units
        self.net  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Dense(out_units,in_units=in_units))#capa de salida
        self.reset(net)
        return net

    def predict(self, par,X):
        yhat=self.forward(par,X_train=X)    
        return yhat	

    def forward(self,par, **args):
        for k,v in args.items():
            if k=='X_train':
                X=v
        y_linear = self.net.forward(X)
        y_hat=mxp.normal.Normal(loc=y_linear,scale=np.sqrt(self.hyper['scale']))
        return y_hat
    
    def reset(self,net,sigma=0.01,init=True):
        net.initialize(init=mx.init.Normal(sigma=sigma), ctx=self.ctx, force_reinit=init)
        return True

    def negative_log_prior(self, par,**args):
        log_prior=np.zeros(shape=1,ctx=self.ctx)
        #param_size=np.sum([par[var].data().size for var in par.keys()])
        param_prior=mxp.normal.Normal(loc=0.,scale=np.sqrt(self.hyper['alpha']))
        for var in par.keys():
            log_prior=log_prior-np.sum(param_prior.log_prob(par[var].data()))
        return log_prior
       
    def negative_log_likelihood(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v
            elif k=='y_train':
                y=v
        y_hat = self.forward(par,X_train=X)
        return -np.sum(y_hat.log_prob(y))
        
    def loss(self,par,**args):
        log_like=self.negative_log_likelihood(par,**args)
        log_prior=self.negative_log_prior(par,**args)
        return log_like+log_prior

class linear_aleatoric(linear):

    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Dense(2*out_units,in_units=in_units))#capa de salida
        self.reset(net)
        return net

    def forward(self,par, **args):
        softplus = lambda x : np.log(1. + np.exp(x))
        for k,v in args.items():
            if k=='X_train':
                X=v
        y_linear = self.net.forward(X)
        scale=1e-3 + softplus(0.05 * y_linear[...,self.out_units:])
        loc=y_linear[..., :self.out_units]
        y_hat=mxp.normal.Normal(loc=loc,scale=scale)
        return y_hat




class lstm_linear(linear):
    
    def __init__(self,_hyper,in_units,n_layers,n_hidden,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,n_layers,n_hidden)
        
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