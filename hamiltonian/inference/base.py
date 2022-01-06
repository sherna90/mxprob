import mxnet.numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy
import mxnet.gluon.probability as mxp
from mxnet.gluon.loss import KLDivLoss
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

    def predict_sample(self,par,data_loader):
        samples=list()
        loglike=list()
        labels=list()
        for X_test,y_test in data_loader:
            X_test=X_test.as_in_context(self.ctx)
            y_test=y_test.as_in_context(self.ctx)
            labels.append(y_test)
            y_pred=self.model.predict(par,X_test)
            if isinstance(y_pred.sample(),mx.numpy.ndarray):
                loglike.append(y_pred.log_prob(y_test.as_np_ndarray()))
            else:
                loglike.append(y_pred.log_prob(y_test))
            samples.append(y_pred.sample())
        samples=np.concatenate(samples)
        labels=np.concatenate(labels)
        loglike=np.concatenate(loglike)
        return samples.asnumpy(),labels.asnumpy(),loglike.asnumpy()

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
        num_batches=n_data/batch_size
        log_likelihood_sum=self.model.negative_log_likelihood(par,**args)
        log_prior_sum=self.model.negative_log_prior_non_centered(par,means,epsilons,sigmas,**args)
        log_var_posterior=np.zeros(shape=1,ctx=self.ctx)
        for var in par.keys():
            l_kl=1.+np.log(np.square(sigmas[var]))
            l_kl=l_kl-np.square(means[var])
            l_kl=l_kl-np.square(sigmas[var])
            log_var_posterior=log_var_posterior-0.5*np.sum(l_kl)
        return 1.0 / n_data * (log_var_posterior + log_prior_sum) + log_likelihood_sum
    
    def distillation_loss(self,par,teacher_predictions,student_predictions,n_data,**args):
        log_likelihood_sum=self.model.negative_log_likelihood(par,**args)
        log_prior_sum=self.model.negative_log_prior(par,**args)
        kl_loss = KLDivLoss(teacher_predictions,student_predictions,from_logits=False)
        return  1.0 / n_data * (kl_loss + log_prior_sum) + log_likelihood_sum
    