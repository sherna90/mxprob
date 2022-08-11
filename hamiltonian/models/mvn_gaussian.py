import warnings
warnings.filterwarnings("ignore")

import mxnet as mx
from mxnet import np
from mxnet import nd, autograd, gluon
import mxnet.gluon.probability as mxp
import hamiltonian.models.model as base_model

class mvn_gaussian:

    def __init__(self,_hyper,in_units,out_units,ctx=mx.cpu()):
        self.hyper=_hyper
        self.ctx=ctx
        self.net  = self._init_net(in_units,out_units)
        
    def _init_net(self,in_units,out_units):
        net = gluon.nn.Sequential()#inicializacion api sequencial
        net.add(gluon.nn.Dense(out_units,in_units=in_units,use_bias=False))#capa de salida
        self.reset(net)
        return net
    
    def reset(self,net,sigma=0.01,init=True):
        net.initialize(init=mx.init.Normal(sigma=sigma), ctx=self.ctx, force_reinit=init)
        return True

    def forward(self,par, **args):
        for k,v in args.items():
            if k=='X_train':
                X=v
        y_linear = self.net.forward(X)
        cov=self.hyper['cov']
        y_hat=mxp.multivariate_normal.MultivariateNormal(loc=y_linear,cov=cov)
        return y_hat

    def predict(self, par,X):
        yhat=self.forward(par,X_train=X)    
        return yhat	

    def negative_log_prior(self, par,**args):
        return 0.

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
        return log_like