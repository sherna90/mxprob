import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet import np,npx
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base
from hamiltonian.inference.sgd import sgd
import mxnet.gluon.probability as mxp
import os
import h5py 
npx.set_np()

class bbb(base):

    def softplus(self,x):
        return np.log(1.0 + np.exp(x))

    def inv_softplus(self,x):
        return np.log(np.exp(x) - 1.0)

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain_name' in args:
            if os.path.exists(args['chain_name']):
                os.remove(args['chain_name'])
            variational_posterior=h5py.File(args['chain_name'],'w')
        else:
            if os.path.exists('variational_posterior.h5'):
                os.remove('variational_posterior.h5')
            variational_posterior=h5py.File('variational_posterior.h5','w')
        params=self.model.net.collect_params()
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        var_params=self.create_variational_params(self.model.net.collect_params(),self.ctx)
        trainer = gluon.Trainer(var_params, 'sgd', {'learning_rate': self.step_size})
        for par in params.values():
            par.grad_req='null'
        for i in range(epochs):
            data_loader,n_batches=self._get_loader(**args)
            cumulative_loss=0
            j=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    sampled_params=self.sample_params(params,var_params)
                    loss=self.variational_loss(sampled_params,var_params,n_batches,batch_size,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus 
                j = j+1 
                cumulative_loss += loss.asnumpy()
                trainer.step(n_batches*batch_size)
            loss_val[i]=cumulative_loss/(n_batches*batch_size)
            #w_0 -= loss_val[i]
            if verbose and (i%(epochs//10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        if variational_posterior:
            dset=[variational_posterior.create_dataset(var,data=params[var].data().asnumpy()) for var in var_params.keys()]     
            variational_posterior.attrs['loss']=loss_val
            variational_posterior.flush()
            variational_posterior.close()
        return params,loss_val,var_params

    def create_variational_params(self,params,context):
        var_params=dict()
        for i, model_param in zip(params,params.values()):
            var_mu = gluon.Parameter(
                'var_mu_{}'.format(i), shape=model_param.shape,
                init=mx.init.Normal(0.))
            var_mu.initialize(ctx=context)
            var_params.update({'var_mu_{}'.format(i):var_mu})

            var_rho = gluon.Parameter(
                'var_rho_{}'.format(i), shape=model_param.shape,
                init=mx.init.Constant(self.inv_softplus(1.)))
            var_rho.initialize(ctx=context)
            var_params.update({'var_rho_{}'.format(i):var_rho})
        return var_params

    def sample_params(self,params,var_params):
        for i,model_param in zip(params,params.values()):
            epsilon=mx.np.random.normal(size=model_param.shape,loc=0.,scale=1.)
            var_sigma=self.softplus(var_params['var_rho_{}'.format(i)].data())
            param_data= var_params['var_mu_{}'.format(i)].data() + var_sigma * epsilon
            model_param.set_data(param_data)
        return params   

    def variational_loss(self,params,var_params,num_batches,batch_size,**args):
        log_likelihood=self.model.negative_log_likelihood(params,**args)
        log_prior=self.model.negative_log_prior(params,**args)
        log_var_posterior=np.zeros(shape=1,ctx=self.ctx)
        for i,model_param in zip(params,params.values()):
            param_scale=self.softplus(var_params['var_rho_{}'.format(i)].data())
            param_location=var_params['var_mu_{}'.format(i)].data()
            param_prior=mxp.normal.Normal(loc=param_location,scale=param_scale)
            log_var_posterior=log_var_posterior-np.sum(param_prior.log_prob(model_param.data()))
        return 1.0 / num_batches * (log_var_posterior + log_prior) + log_likelihood

    def predict(self,means,sigmas,num_samples,**args):
        data_loader,_=self._get_loader(**args)
        total_samples=[]
        total_loglike=[]
        total_labels=[]
        posterior=dict()
        for var in means.keys():
            variational_posterior=mxp.normal.Normal(loc=means[var],
                                            scale=sigmas[var])
            posterior.update({var:variational_posterior})
        for i in range(num_samples):
            samples=[]
            labels=[]
            loglike=[]
            par=dict()
            for var in means.keys():
                par.update({var:posterior[var].sample().as_nd_ndarray()})
                #par.update({var:means[var].as_nd_ndarray()})
            for X_test,y_test in data_loader:
                X_test=X_test.as_in_context(self.ctx)
                y_test=y_test.as_in_context(self.ctx)
                y_pred=self.model.predict(par,X_test)
                if isinstance(y_pred.sample(),mx.numpy.ndarray):
                    loglike.append(y_pred.log_prob(y_test.as_np_ndarray()).asnumpy())
                else:
                    loglike.append(y_pred.log_prob(y_test).asnumpy())
                samples.append(y_pred.sample().asnumpy())
                labels.append(y_test.asnumpy())
            total_samples.append(np.concatenate(samples))
            total_loglike.append(np.concatenate(loglike))
        total_labels=np.concatenate(labels)
        total_samples=np.stack(total_samples)
        total_loglike=np.stack(total_loglike)
        return total_samples,total_labels,total_loglike