import warnings
warnings.filterwarnings("ignore")

import mxnet as mx
from mxnet import np,npx
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp

from mxnet.gluon.model_zoo.vision import resnet,vgg 
npx.set_np()               

class softmax():
    
    def __init__(self,_hyper,in_units,out_units,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net,self.par  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(out_units,in_units=in_units[1]*in_units[2]))#capa de salida
        par=self.reset(net)
        return net,par

    def reset(self,net,init=True):
        if init:
            net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx, force_reinit=True)
        par=dict()
        for name,gluon_par in net.collect_params().items():
            par.update({name:gluon_par.data()})
            #gluon_par.grad_req='null'
        return par

    def softmax(self, y_linear):
        exp = np.exp(y_linear-np.max(y_linear, axis=1).reshape((-1,1)))
        norms = np.sum(exp, axis=1).reshape((-1,1))
        return exp / norms

    def predict(self, par,X,prob=False):
        y_hat=self.forward(par,X_train=X)   
        return y_hat	

    def forward(self,par, **args):
        for k,v in args.items():
            if k=='X_train':
                X=v.as_in_context(self.ctx)
        for name,gluon_par in self.net.collect_params().items():
            gluon_par.set_data(par[name])
        y_linear = self.net.forward(X)
        yhat = self.softmax(y_linear)
        cat=mxp.Categorical(1,prob=yhat)
        return cat
     
    def negative_log_prior(self, par,**args):
        log_prior=np.zeros(shape=1,ctx=self.ctx)
        for var in par.keys():
            means=np.zeros(par[var].shape,ctx=self.ctx)
            sigmas=np.ones(par[var].shape,ctx=self.ctx)*np.sqrt(self.hyper['alpha'])
            param_prior=mxp.normal.Normal(loc=means,scale=sigmas)
            #theta=nd.array(par[var]).as_in_context(self.ctx)
            log_prior=log_prior-np.sum(param_prior.log_prob(par[var]))
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

    def negative_log_prior_non_centered(self,par, means,epsilons,stds,**args):
        log_prior=np.zeros(shape=1,ctx=self.ctx)
        scale_prior=mxp.Normal(loc=0.0,scale=1.0)
        location_prior=mxp.Normal(loc=0.0,scale=1.0)
        epsilons_prior=mxp.Normal(loc=0.0,scale=1.0)
        for var in means.keys():
            #log_sigmas=nd.log(stds[var].as_nd_ndarray())
            log_prior=log_prior-np.sum(scale_prior.log_prob(stds[var]))
            log_prior=log_prior-np.sum(epsilons_prior.log_prob(epsilons[var]))
            log_prior=log_prior-np.sum(location_prior.log_prob(means[var]))
        return log_prior

    def negative_log_prior_centered(self,par, means,epsilons,stds,**args):
        log_prior=nd.zeros(shape=1,ctx=self.ctx)
        scale_prior=mxp.Normal(loc=0.0,scale=1.0)
        location_prior=mxp.Normal(loc=0.0,scale=1.0)
        for var in means.keys():
            theta_prior=mxp.Normal(loc=means[var],scale=stds[var])
            #log_sigmas=nd.log(stds[var].as_nd_ndarray())
            log_prior=log_prior-nd.nansum(scale_prior.log_prob(stds[var]).as_nd_ndarray())
            log_prior=log_prior-nd.nansum(location_prior.log_prob(means[var]).as_nd_ndarray())
            log_prior=log_prior-nd.nansum(theta_prior.log_prob(par[var]).as_nd_ndarray())
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
    
    def __init__(self,_hyper,in_units=(1,32,32),out_units=10,n_layers=18,pre_trained=False,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.version=2
        #n_layers = 18, 34, 50, 101, 152.
        self.pre_trained=pre_trained
        self.net,self.par  = self._init_net(in_units,out_units,n_layers)
        
    def _init_net(self,in_units,out_units,n_layers):
        data = nd.numpy.ones((1,in_units[0],in_units[1],in_units[2]))
        net = gluon.nn.HybridSequential()#inicializacion api sequencial
        model=resnet.get_resnet(self.version,n_layers,pretrained=self.pre_trained,ctx=self.ctx)
        net.add(model.features)
        net.add(gluon.nn.Dense(out_units))#capa de salida
        net[1].initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        net(data.as_in_context(self.ctx))
        net.hybridize()
        par=self.reset(net,init=False)
        return net,par



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
        data = mx.np.ones((1,in_units[0],in_units[1],in_units[2]))
        net(data.as_in_context(self.ctx))
        #net.hybridize()
        par=self.reset(net)
        return net,par
    

class vgg_softmax(softmax):
    
    def __init__(self,_hyper,in_units=(3,256,256),out_units=2,n_layers=16,pre_trained=True,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        #n_layers = 11, 13, 16, 19.
        self.pre_trained=pre_trained
        self.net,self.par  = self._init_net(in_units,out_units,n_layers)
        
    def _init_net(self,in_units,out_units,n_layers):
        data = nd.ones((1,in_units[0],in_units[1],in_units[2]))
        net = gluon.nn.Sequential()
        model=vgg.get_vgg(n_layers, pretrained=self.pre_trained, ctx=self.ctx)
        net.add(model.features)
        net.add(gluon.nn.Dense(out_units))
        net(data.as_in_context(self.ctx))
        net[1].initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx)
        par=self.reset(net,init=False)
        return net,par

