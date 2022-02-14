import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.ndarray import clip
from mxnet.gluon.metric import Accuracy,RMSE
from tqdm import tqdm, trange
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
        loss_val=np.zeros(epochs)
        params=self.model.net.collect_params()
        data_loader,n_batches=self._get_loader(**args)
        j=0
        trainer = gluon.Trainer(params, 'sgd', {'learning_rate': self.step_size})
        for i in range(epochs):
            cumulative_loss=list()
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    loss = self.loss(params,X_train=X_batch,y_train=y_batch,n_data=n_batches*batch_size)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                trainer.step(batch_size)
                y_pred=self.model.predict(params,X_batch)
                metric.update(y_batch, np.mean(y_pred.sample_n(100),axis=0))
                cumulative_loss.append(loss.asnumpy())
            _,train_accuracy=metric.get()
            loss_val[i]=loss
            if verbose and i%(epochs//10)==0:
                print('iteration {0}, train loss: {1:.4f}, train accuracy : {2:.4f}'.format(i,loss_val[i],train_accuracy))
        dset=[posterior_samples.create_dataset(var,data=params[var].data().asnumpy()) for var in params.keys()]
        posterior_samples.attrs['epochs']=epochs
        posterior_samples.attrs['loss']=loss_val
        posterior_samples.flush()
        posterior_samples.close()
        return params,loss_val


    def step(self,momentum,params):
        for var,par in zip(params,params.values()):
            grad=par.grad()
            momentum[var] = self.gamma * momentum[var] + self.step_size * grad #calcula para parametros peso y bias
            par.data()[:]=par.data()- self.step_size * grad
        return momentum,params

    def predict(self,par,num_samples=100,**args):
        data_loader,n_examples=self._get_loader(**args)
        total_labels=[]
        total_samples=[]
        total_loglike=[]
        params=self.model.net.collect_params()
        for var,theta in zip(params,params.values()):
            if var in par:
                theta.data()[:]=mx.numpy.array(par[var]).copyto(self.ctx)
        for X_test,y_test in data_loader:
            X_test=X_test.as_in_context(self.ctx)
            y_test=y_test.as_in_context(self.ctx)
            y_hat=self.model.predict(par,X_test)
            total_loglike.append(y_hat.log_prob(y_test).asnumpy())
            samples=[]
            for _ in range(num_samples):
                samples.append(y_hat.sample().asnumpy())
            total_samples.append(samples)
            total_labels.append(y_test.asnumpy())
        total_samples=np.concatenate(total_samples,axis=1)
        total_labels=np.concatenate(total_labels)
        total_loglike=np.concatenate(total_loglike)
        return total_samples,total_labels,total_loglike   