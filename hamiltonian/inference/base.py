import mxnet.numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy
import mxnet.gluon.probability as mxp

class base:

    def __init__(self,model,step_size=0.1,ctx=mx.cpu()):
        self.step_size = step_size
        self.model = model
        self.ctx=ctx
        self.gamma=0.9
        
    def _get_loader(self,**args):
        data_loader=None
        n_examples=0
        if 'X_train' in args and 'y_train' in args:
            X=args['X_train']
            y=args['y_train']
            n_examples=X.shape[0]
            batch_size=64
            data_loader=self.iterate_minibatches(X, y,batch_size)
        elif 'data_loader' in args:
            data_loader=args['data_loader']
            n_examples=len(data_loader)
        return data_loader,n_examples

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield np.array(X[excerpt],ctx=self.ctx), nd.array(y[excerpt],ctx=self.ctx)#generador, pasa los datos originales a la memoria donde se este trabajando cpu o gpu
            #devuelve un arreglo mxnet
    
    def step(self,batch_size,momentum,par):
        pass

    def fit(self,epochs=1,batch_size=1,**args):
        pass

    def lr_schedule(self,initial_step_size,step,decay_factor,num_batches):
        return initial_step_size * (1.0/(1.0+step*decay_factor*num_batches))
        #return initial_step_size - (decay_factor*step/num_batches)

    def loss(self,par,X_train,y_train):
        batch_size=X_train.shape[0]
        return self.model.loss(par,X_train=X_train,y_train=y_train)*1./batch_size
    
    def centered_hierarchical_loss(self,par,means,epsilons,stds,**args):
        log_like=self.model.negative_log_likelihood(par,**args)
        log_prior=self.model.negative_log_prior_centered(par,means,epsilons,stds,**args)
        return log_like+log_prior

    def non_centered_hierarchical_loss(self,par,means,epsilons,stds,**args):
        log_like=self.model.negative_log_likelihood(par,**args)
        log_prior=self.model.negative_log_prior_non_centered(par,means,epsilons,stds,**args)
        return log_like+log_prior

    def variational_loss(self,par,means,epsilons,sigmas,n_data,batch_size,**args):
        for k,v in args.items():
            if k=='X_train':
                X_train=v
            elif k=='y_train':
                y_train=v
        num_batches=n_data/batch_size
        #for var in self.model.par.keys():
        #    if type(self.model.par[var])=='mxnet.numpy.ndarray':
        #        par.update({var:par[var].as_nd_ndarray()})
        log_likelihood_sum=self.model.negative_log_likelihood(par,X_train=X_train,y_train=y_train)
        log_prior_sum=self.model.negative_log_prior_non_centered(par,means,epsilons,sigmas,X_train=X_train,y_train=y_train)
        log_var_posterior=np.zeros(shape=1,ctx=self.ctx)
        for var in par.keys():
            l_kl=1.+np.log(np.square(sigmas[var]))
            l_kl=l_kl-np.square(means[var])
            l_kl=l_kl-np.square(sigmas[var])
            log_var_posterior=log_var_posterior-0.5*np.sum(l_kl)
        return 1.0 / n_data * (log_var_posterior + log_prior_sum) + log_likelihood_sum
    