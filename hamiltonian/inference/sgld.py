from mxnet.ndarray.contrib import isnan
import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon,random
from mxnet.gluon.metric import Accuracy
from mxnet.ndarray import clip
import mxnet.gluon.probability as mxp
from copy import deepcopy
from hamiltonian.inference.base import base
from hamiltonian.inference.sgd import sgd
import h5py
import os 

class sgld(base):

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain' in args:
            chain=args['chain']
        else:
            chain=None
        if 'dataset' in args:
            dataset=args['dataset']
        else:
            dataset=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        params=self.model.net.collect_params()
        momentum={var:nd.numpy.zeros_like(params[var].data()) for var in params.keys()} #single_chain={var:list() for var in par.keys()}
        epsilon=self.step_size
        j=0
        data_loader,n_examples=self._get_loader(**args)
        if 'valid_data_loader' in args:
            val_data_loader=args['valid_data_loader']
        else:
            val_data_loader=None
        accuracy=Accuracy()
        schedule = mx.lr_scheduler.FactorScheduler(step=250, factor=0.5)
        schedule.base_lr = self.step_size
        iteration_idx=1
        for i in range(epochs):
            cumulative_loss=list()
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    loss = self.loss(params,X_train=X_batch,y_train=y_batch,n_data=n_examples)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                self.step_size = schedule(iteration_idx)
                iteration_idx += 1
                momentum,params=self.step(momentum,params,n_data=n_examples)
                y_pred=self.model.predict(params,X_batch)
                accuracy.update(y_batch, y_pred.sample())
                cumulative_loss.append(loss)
                j=j+1
            loss_val[i]=np.sum(cumulative_loss)/n_examples
            if dataset:
                for var in params.keys():
                    dataset[var][chain,i,:]=params[var].data().asnumpy()
            if verbose:
                _,train_accuracy=accuracy.get()
                print('iteration {0}, train loss: {1:.4f}, train accuracy : {2:.4f}'.format(i,loss_val[i],train_accuracy))
        return params,loss_val
    
    def step(self,momentum,params,n_data=1.):
        normal=self.draw_momentum(params,self.step_size)
        for var,par in zip(params,params.values()):
            try:
                grad=par.grad()
                momentum[var][:] = self.gamma*momentum[var]+ (1.-self.gamma)*nd.np.square(grad)
                par.data()[:]=par.data()-0.5*self.step_size*grad/nd.np.sqrt(momentum[var] + 1e-6)+normal[var]*1./n_data
            except:
                None
        return momentum, params

    def draw_momentum(self,params,epsilon):
        momentum={var:random.normal(0,np.sqrt(epsilon),
            shape=params[var].shape,
            ctx=self.ctx,
            dtype=params[var].dtype) for var in params.keys()}
        return momentum

    def sample(self,epochs=1,batch_size=1,chains=2,verbose=False,**args):
        loss_values=list()                                                                                                              
        posterior_samples_multiple_chains=list()
        if 'chain_name' in args:
            if os.path.exists(args['chain_name']):
                os.remove(args['chain_name'])
            posterior_samples=h5py.File(args['chain_name'],'w')
        else:
            if os.path.exists('posterior_samples.h5'):
                os.remove('posterior_samples.h5')
            posterior_samples=h5py.File('posterior_samples.h5','w')
        params=self.model.net.collect_params()
        dset=[posterior_samples.create_dataset(var,shape=(chains,epochs)+params[var].shape,dtype=params[var].dtype) for var in params.keys()]
        for i in range(chains):
            self.model.reset(self.model.net,sigma=1e-3,init=False)
            _,loss=self.fit(epochs=epochs,batch_size=batch_size,
                chain=i,dataset=posterior_samples,verbose=verbose,**args)
            loss_values.append(loss)
        posterior_samples.attrs['num_chains']=chains
        posterior_samples.attrs['num_samples']=epochs 
        posterior_samples.attrs['loss']=np.stack(loss_values) 
        posterior_samples.flush()
        posterior_samples.close()  
        return posterior_samples,loss_values

    def predict(self,posterior_samples,**args):
        data_loader,_=self._get_loader(**args)
        total_samples=list()
        num_samples=posterior_samples.attrs['num_samples']
        num_chains=posterior_samples.attrs['num_chains']
        total_loglike=list()
        params=self.model.net.collect_params()
        for i in range(num_chains):
            for j in range(num_samples):
                par=dict()
                for var in posterior_samples.keys():
                    par.update({var:mx.np.array(posterior_samples[var][i,j,:],ctx=self.ctx)})
                for var,theta in zip(params,params.values()):
                    if var in par:
                        theta.data()[:]=par[var]
                samples=list()
                loglike=list()
                labels=list()
                for X_test,y_test in data_loader:
                    X_test=X_test.as_in_context(self.ctx)
                    y_test=y_test.as_in_context(self.ctx)
                    labels.append(y_test.asnumpy())
                    y_pred=self.model.predict(par,X_test)
                    if isinstance(y_pred.sample(),mx.numpy.ndarray):
                        loglike.append(y_pred.log_prob(y_test.as_np_ndarray()).asnumpy())
                    else:
                        loglike.append(y_pred.log_prob(y_test).asnumpy())
                    samples.append(y_pred.sample().asnumpy())
                total_samples.append(np.concatenate(samples))
                total_loglike.append(np.concatenate(loglike))
        total_samples=np.stack(total_samples)
        total_loglike=np.stack(total_loglike)
        total_labels=np.concatenate(labels)
        return total_samples,total_labels,total_loglike


    def posterior_diagnostics(self,posterior_samples):
        num_chains=posterior_samples.attrs['num_chains']
        posterior_samples_multiple_chains=list()
        for i in range(num_chains):
            posterior_samples_single_chain={var:posterior_samples[var][:] for var in self.model.par}
            posterior_samples_multiple_chains.append(posterior_samples_single_chain)
        posterior_samples_multiple_chains_expanded=[ {var:np.expand_dims(sample,axis=0) for var,sample in posterior.items()} for posterior in posterior_samples_multiple_chains]
        return posterior_samples_multiple_chains_expanded


class hierarchical_sgld(sgld):
    
    def softplus(self,x):
        return nd.np.log(1. + nd.np.exp(x))
        #return nd.exp(x)

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain' in args:
            chain=args['chain']
        else:
            chain=None
        if 'dataset' in args:
            dataset=args['dataset']
        else:
            dataset=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        params=self.model.net.collect_params()

        scale_prior=mxp.Normal(loc=0.,scale=1.0)
        means_prior=mxp.Normal(loc=0.,scale=1.0)
        eps_prior=mxp.Normal(loc=0.,scale=1.0)
        
        mean_momentum={var:nd.numpy.zeros_like(params[var].data()).copyto(self.ctx) for var in params.keys()} #single_chain={var:list() for var in par.keys()}
        std_momentum={var:nd.numpy.zeros_like(params[var].data()).copyto(self.ctx) for var in params.keys() } #single_chain={var:list() for var in par.keys()}
        
        stds={var:scale_prior.sample(params[var].shape).copyto(self.ctx) for var in params.keys() }
        means={var:params[var].data() for var in params.keys() }
        for (m,s,p) in zip(means.values(),stds.values(),params.values()):
            m.attach_grad()
            s.attach_grad()
            p.grad_req='null'
        accuracy=Accuracy()
        data_loader,n_examples=self._get_loader(**args)
        for i in range(epochs):
            cumulative_loss=list()
            j=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                epsilons={var:eps_prior.sample(params[var].shape).copyto(self.ctx) for var in params.keys()}            
                sigmas={var:nd.numpy.zeros_like(means[var],ctx=self.ctx) for var in means.keys()}
                with autograd.record():
                    for var in means.keys():
                        sigmas[var][:]=self.softplus(stds[var])
                        params[var].data()[:]=means[var][:]+epsilons[var][:]*sigmas[var][:]
                    loss=self.centered_hierarchical_loss(params,means,epsilons,sigmas,X_train=X_batch,y_train=y_batch,n_data=n_examples)
                loss.backward()
                y_pred=self.model.predict(params,X_batch)
                accuracy.update(y_batch, y_pred.sample())
                mean_momentum, means = self.step(mean_momentum, means)
                std_momentum, stds = self.step(std_momentum, stds)
                j = j+1 
                cumulative_loss.append(loss)
            loss_val[i]=np.sum(cumulative_loss)/n_examples
            if dataset:
                for var in params.keys():
                    dataset[var][chain,i,:]=params[var].data().asnumpy()
            if verbose:
                _,train_accuracy=accuracy.get()
                print('iteration {0}, train loss: {1:.4f}, train accuracy : {2:.4f}'.format(i,loss_val[i],train_accuracy))
        return params,loss_val

class distillation_sgld(sgld):
    
    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain' in args:
            chain=args['chain']
        else:
            chain=None
        if 'dataset' in args:
            dataset=args['dataset']
        else:
            dataset=None
        if 'teacher' in args:
            teacher=args['teacher']
        else:
            teacher=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        params=self.model.net.collect_params()
        momentum={var:nd.numpy.zeros_like(params[var].data()) for var in params.keys()} #single_chain={var:list() for var in par.keys()}
        j=0
        data_loader,n_examples=self._get_loader(**args)
        if 'valid_data_loader' in args:
            val_data_loader=args['valid_data_loader']
        else:
            val_data_loader=None
        accuracy=Accuracy()
        schedule = mx.lr_scheduler.FactorScheduler(step=250, factor=0.5)
        schedule.base_lr = self.step_size
        distillation_loss = mx.gluon.loss.KLDivLoss(from_logits=False)
        iteration_idx=1
        teacher.net.setattr('grad_req', 'null')
        for i in range(epochs):
            cumulative_loss=list()
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    student_loss = self.loss(params,X_train=X_batch,y_train=y_batch,n_data=n_examples)
                    student_predictions = self.model.forward(params,X_train=X_batch)
                    with autograd.predict_mode():
                        teacher_predictions = teacher.forward(teacher.net.collect_params(),X_train=X_batch.as_in_context(teacher.ctx))
                    teacher_loss = distillation_loss(teacher_predictions.prob.as_in_context(self.ctx),student_predictions.prob).sum()
                    loss = self.gamma*student_loss + (1.-self.gamma)*teacher_loss
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                self.step_size = schedule(iteration_idx)
                iteration_idx += 1
                momentum,params=self.step(momentum,params,n_data=n_examples)
                accuracy.update(y_batch, student_predictions.sample())
                cumulative_loss.append(loss)
                j=j+10
            loss_val[i]=np.sum(cumulative_loss)/n_examples
            if dataset:
                for var in params.keys():
                    dataset[var][chain,i,:]=params[var].data().asnumpy()
            if verbose:
                _,train_accuracy=accuracy.get()
                print('iteration {0}, train loss: {1:.4f}, train accuracy : {2:.4f}'.format(i,loss_val[i],train_accuracy))
        return params,loss_val