import mxnet as mx
from mxnet import nd,np, autograd, gluon
from mxnet.ndarray import clip
from mxnet.gluon.metric import Accuracy,RMSE
from mxnet.gluon.utils import split_and_load
from copy import deepcopy
from hamiltonian.inference.base import base
import h5py 
import os 

class sgd(base):

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain_name' in args:
            if os.path.exists(args['chain_name']):
                os.remove(args['chain_name'])
            posterior_samples=h5py.File(args['chain_name'],'w')
        else:
            if os.path.exists('map_estimate.h5'):
                os.remove('map_estimate.h5')
            posterior_samples=h5py.File('map_estimate.h5','w')
        if 'metric' in args:
            if args['metric']=='rmse':
                metric=RMSE()
            elif args['metric']=='accuracy':
                metric=Accuracy()
        else:
            metric=Accuracy()
        epochs=int(epochs)
        loss_val=list()
        params=self.model.net.collect_params()
        data_loader,n_batches=self._get_loader(**args)
        momentum={var:mx.np.zeros_like(params[var].data()) for var in params.keys()}
        #trainer = gluon.Trainer(params, 'sgd', {'learning_rate': self.step_size})
        for i in range(epochs):
            cumulative_loss=0.0
            for j,(X_batch, y_batch) in enumerate(data_loader):
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    #loss = self.loss(params,X_train=X_batch,y_train=y_batch,n_data=n_batches*batch_size)
                    loss=self.model.loss(params,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                #trainer.step(batch_size)
                cumulative_loss+=loss.asnumpy()
                self.step(momentum,params)
                y_pred=self.model.predict(params,X_batch)
                metric.update(labels=[y_batch], preds=[mx.np.quantile(y_pred.sample_n(100),.5,axis=0).astype(y_batch.dtype)])    
            metric_name,train_accuracy=metric.get()
            loss_val.append(cumulative_loss/(n_batches*batch_size))
            print('iteration {0}, train loss: {1:.4f}, train {2} : {3:.4f}'.format(i,loss_val[-1],metric_name,train_accuracy))
        dset=[posterior_samples.create_dataset(var,data=params[var].data().asnumpy()) for var in params.keys()]
        posterior_samples.attrs['epochs']=epochs
        posterior_samples.attrs['loss']=loss_val
        posterior_samples.flush()
        posterior_samples.close()
        return params,loss_val

    def step(self,momentum,params):
        for var in params:
            try:
                grad=params[var].grad()
                momentum[var] = self.gamma*momentum[var]+ (1.-self.gamma)*np.square(grad)
                params[var].data()[:]=params[var].data()-0.5*self.step_size*grad/np.sqrt(momentum[var] + 1e-6)
            except:
                None

    def predict(self,par,num_samples=100,**args):
        data_loader,n_examples=self._get_loader(**args)
        total_labels=[]
        total_samples=[]
        total_loglike=[]
        params=self.model.net.collect_params()
        for var in params:
            if var in par:
                params[var].data()[:]=mx.numpy.array(par[var]).copyto(self.ctx)
        for X_test,y_test in data_loader:
            X_test=X_test.as_in_context(self.ctx)
            y_test=y_test.as_in_context(self.ctx)
            y_hat=self.model.predict(par,X_test)
            total_loglike.append(y_hat.log_prob(y_test))
            total_samples.append(y_hat.sample_n(num_samples))
            total_labels.append(y_test)
        total_samples=np.concatenate(total_samples,axis=1)
        total_labels=np.concatenate(total_labels)
        #total_loglike=np.concatenate(total_loglike)
        return total_samples,total_labels,total_loglike   

class sgd_multi_gpu(base):

    def allreduce(self,data):
        for i in range(1, len(data)):
            data[0][:] += data[i].copyto(data[0].ctx)
        for i in range(1, len(data)):
            data[0].copyto(data[i])

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain_name' in args:
            if os.path.exists(args['chain_name']):
                os.remove(args['chain_name'])
            posterior_samples=h5py.File(args['chain_name'],'w')
        else:
            if os.path.exists('map_estimate.h5'):
                os.remove('map_estimate.h5')
            posterior_samples=h5py.File('map_estimate.h5','w')
        if 'metric' in args:
            if args['metric']=='rmse':
                metric=RMSE()
            elif args['metric']=='accuracy':
                metric=Accuracy()
        else:
            metric=Accuracy()
        epochs=int(epochs)
        loss_val=list()
        params=self.model.net.collect_params()
        data_loader,n_batches=self._get_loader(**args)
        momentum={var:[mx.np.zeros_like(par) for par in params[var].list_data() ] for var in params.keys()}
        for i in range(epochs):
            cumulative_loss=0.0
            for j,(X_batch, y_batch) in enumerate(data_loader):
                y_list=split_and_load(y_batch,self.ctx)
                X_list=split_and_load(X_batch,self.ctx)
                with autograd.record():
                    #loss = [self.loss(params,X_train=data,y_train=label,n_data=n_batches*batch_size/len(self.ctx)) for data,label in zip(X_list,y_list)]
                    loss = [self.model.loss(params,X_train=data,y_train=label) for data,label in zip(X_list,y_list)]
                for l in loss:
                    l.backward()
                for var in params:
                    self.allreduce(params[var].list_grad())
                cumulative_loss+=sum([l.asnumpy() for l in loss])
                momentum,params=self.step(momentum,params)
                for (data,label) in zip(X_list,y_list):
                    y_pred=self.model.predict(params,data)
                    metric.update(labels=[label.as_in_context(mx.cpu())], preds=[mx.np.quantile(y_pred.sample_n(100),.5,axis=0).astype(label.dtype).as_in_context(mx.cpu())])   
                #print('prediction done!')
            metric_name,train_accuracy=metric.get()
            loss_val.append(cumulative_loss/(n_batches*batch_size))
            print('iteration {0}, train loss: {1:.4f}, train {2} : {3:.4f}'.format(i,loss_val[-1],metric_name,train_accuracy))
        dset=[posterior_samples.create_dataset(var,data=params[var].list_data()[0].asnumpy()) for var in params.keys()]
        posterior_samples.attrs['epochs']=epochs
        posterior_samples.attrs['loss']=loss_val
        posterior_samples.flush()
        posterior_samples.close()
        return params,loss_val

    def step(self,momentum,params):
        for var,par in zip(params,params.values()):
            for theta,rho in zip(par.list_data(),momentum[var]):
                try:
                    grad=theta.grad
                    rho[:] = self.gamma*rho+ (1.-self.gamma)*nd.np.square(grad)
                    theta[:]=theta-0.5*self.step_size*grad/nd.np.sqrt(rho + 1e-6)
                except:
                    None
        return momentum, params

    def predict(self,par,num_samples=100,**args):
        data_loader,n_examples=self._get_loader(**args)
        total_labels=[]
        total_samples=[]
        total_loglike=[]
        params=self.model.net.collect_params()
        for var in params:
            if var in par:
                for theta in params[var].list_data():
                    theta[:]=mx.numpy.array(par[var]).copyto(theta.ctx)
        for X_test,y_test in data_loader:
            X_list=split_and_load(X_test,self.ctx)
            y_list=split_and_load(y_test,self.ctx)
            for (data,label) in zip(X_list,y_list):
                y_hat=self.model.predict(params,data)
                total_loglike.append(y_hat.log_prob(label).as_in_context(mx.cpu()))
                total_samples.append(y_hat.sample_n(num_samples).as_in_context(mx.cpu()))
                total_labels.append(label.as_in_context(mx.cpu()))
        total_samples=np.concatenate(total_samples,axis=1)
        total_labels=np.concatenate(total_labels)
        #total_loglike=np.concatenate(total_loglike)
        return total_samples,total_labels,total_loglike 