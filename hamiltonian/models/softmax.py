import warnings
warnings.filterwarnings("ignore")

import mxnet as mx
from mxnet import np,npx
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp

from mxnet.gluon.model_zoo.vision import get_model
npx.set_np()               

class softmax():
    
    def __init__(self,_hyper,in_units,out_units,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Flatten())
        net.add(gluon.nn.Dense(out_units,in_units=in_units[1]*in_units[2]))#capa de salida
        self.reset(net)
        return net

    def reset(self,net,init=True):
        net.initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx, force_reinit=init)
        return True

    def predict(self,par,X):
        y_hat=self.forward(par,X_train=X)   
        return y_hat	

    def forward(self,par,**args):
        for k,v in args.items():
            if k=='X_train':
                X=v.as_in_context(self.ctx)
        [v.set_data(par[u]) for u,v in zip(self.net.collect_params(),self.net.collect_params().values())]      
        y_linear = self.net.forward(X)
        yhat = npx.softmax(y_linear)
        cat=mxp.Categorical(1,prob=yhat)
        return cat
     
    def negative_log_prior(self, par,**args):
        log_prior=np.zeros(shape=1,ctx=self.ctx)
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
        log_prior=np.zeros(shape=1,ctx=self.ctx)
        scale_prior=mxp.Normal(loc=0.0,scale=1.0)
        location_prior=mxp.Normal(loc=0.0,scale=1.0)
        for var in means.keys():
            theta_prior=mxp.Normal(loc=means[var],scale=stds[var])
            #log_sigmas=nd.log(stds[var].as_nd_ndarray())
            log_prior=log_prior-np.sum(scale_prior.log_prob(stds[var]))
            log_prior=log_prior-np.sum(location_prior.log_prob(means[var]))
            log_prior=log_prior-np.sum(theta_prior.log_prob(par[var].data()))
        return log_prior


class pretrained_model(softmax):
    
    def __init__(self,model_name,_hyper,in_units=(1,224,224),out_units=10,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.model_name=model_name
        self.net = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        data = nd.numpy.ones((1,in_units[0],in_units[1],in_units[2]))
        net = gluon.nn.HybridSequential()#inicializacion api sequencial
        model=get_model(self.model_name,pretrained=True,ctx=self.ctx)
        net.add(model.features)
        net.add(gluon.nn.Dense(out_units))#capa de salida
        self.reset(net)
        net(data.as_in_context(self.ctx))
        net.hybridize()
        return net

    def reset(self,net,init=True):
        net[1].initialize(init=mx.init.Normal(sigma=0.01), ctx=self.ctx, force_reinit=init)
        return True

class lenet(softmax):
    
    def __init__(self,_hyper,in_units=(1,28,28),out_units=10,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.HybridSequential()#inicializacion api sequencial
        net.add(
            gluon.nn.Conv2D(channels=6, kernel_size=5, padding=2, activation='softrelu'),
            gluon.nn.AvgPool2D(pool_size=2, strides=2),
            gluon.nn.Conv2D(channels=16, kernel_size=5, activation='softrelu'),
            gluon.nn.AvgPool2D(pool_size=2, strides=2),
            gluon.nn.Dense(120, activation='sigmoid'), 
            gluon.nn.Dense(84, activation='sigmoid'))
        net.add(gluon.nn.Dense(out_units))#capa de salida
        self.reset(net)
        data = mx.np.ones((1,in_units[0],in_units[1],in_units[2]))
        net(data.as_in_context(self.ctx))
        net.hybridize()
        return net
    



