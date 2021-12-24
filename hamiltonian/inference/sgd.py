import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.ndarray import clip
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
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        if 'chain_name' in args:
            if os.path.exists(args['chain_name']):
                os.remove(args['chain_name'])
            posterior_samples=h5py.File(args['chain_name'],'w')
        else:
            if os.path.exists('map_estimate.h5'):
                os.remove('map_estimate.h5')
            posterior_samples=h5py.File('map_estimate.h5','w')
        params=self.model.net.collect_params()
        momentum={var:nd.numpy.zeros_like(params[var].data()) for var in params.keys()}
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            j=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    loss = self.loss(params,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                momentum,params=self.step(batch_size,momentum,params)
                cumulative_loss += nd.numpy.mean(loss)
            loss_val[i]=cumulative_loss/n_examples
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        dset=[posterior_samples.create_dataset(var,data=params[var].data().asnumpy()) for var in params.keys()]
        posterior_samples.attrs['epochs']=epochs
        posterior_samples.attrs['loss']=loss_val
        posterior_samples.flush()
        posterior_samples.close()
        return params,loss_val

    def acc(output, label):
        # output: (batch, num_output) float32 ndarray
        # label: (batch, ) int32 ndarray
        return (output.argmax(axis=1) ==label.astype('float32')).mean().asscalar()

    def step(self,batch_size,momentum,params):
        for var,par in zip(params,params.values()):
            grad=par.grad()
            momentum[var] = self.gamma * momentum[var] + self.step_size * grad #calcula para parametros peso y bias
            par.data()[:]=par.data()-momentum[var]
        return momentum,params

    def predict(self,num_samples=100,**args):
        data_loader,n_examples=self._get_loader(**args)
        total_labels=[]
        total_samples=[]
        total_loglike=[]
        for X_test,y_test in data_loader:
            X_test=X_test.as_in_context(self.ctx)
            y_test=y_test.as_in_context(self.ctx)
            y_hat=self.model.predict(X_test)
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