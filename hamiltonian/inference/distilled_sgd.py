import numpy as np
import mxnet as mx
from mxnet import nd,npx, autograd, gluon
from mxnet.ndarray import clip
from mxnet.gluon.metric import Accuracy,RMSE
from mxnet.gluon.loss import KLDivLoss
from copy import deepcopy
from hamiltonian.inference.base import base
from hamiltonian.inference.sgd import sgd
import h5py 
import os 

class distilled_sgd(sgd):

    def __init__(self,teacher,student,alpha=0.9,temperature=0.9,step_size=0.1,ctx=mx.cpu()):
        self.step_size = step_size
        self.teacher = teacher
        self.student = student
        self.alpha=alpha
        self.temperature=temperature
        self.distillation_loss_fn=KLDivLoss(from_logits=True, axis=-1, weight=None, batch_axis=0)
        self.ctx=ctx
        self.gamma=0.9

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
        params=self.student.net.collect_params()
        teacher_params=self.teacher.net.collect_params()
        data_loader,n_batches=self._get_loader(**args)
        j=0
        momentum={var:mx.np.zeros_like(params[var].data()) for var in params.keys()}
        for i in range(epochs):
            cumulative_loss=list()
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                teacher_predictions = self.teacher.forward(teacher_params,X_train=X_batch).logit
                with autograd.record():
                    student_predictions=self.student.forward(params,X_train=X_batch).logit
                    student_loss = self.student.loss(params,X_train=X_batch,y_train=y_batch,n_data=n_batches)
                    distillation_loss = self.distillation_loss_fn(
                        npx.softmax(teacher_predictions / self.temperature, axis=1),
                        npx.softmax(student_predictions / self.temperature, axis=1))
                    loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                #trainer.step(batch_size)
                momentum,params=self.step(momentum,params)
                y_pred=self.student.predict(params,X_batch)
                metric.update(labels=[y_batch], preds=[mx.np.quantile(y_pred.sample_n(100),.5,axis=0).astype(y_batch.dtype)])
                cumulative_loss.append(loss.asnumpy())
            metric_name,train_accuracy=metric.get()
            loss_val[i]=np.sum(cumulative_loss)/(n_batches*batch_size)
            if verbose and i%(epochs//10)==0:
                print('iteration {0}, train loss: {1:.4f}, train {2} : {3:.4f}'.format(i,loss_val[i],metric_name,train_accuracy))
        dset=[posterior_samples.create_dataset(var,data=params[var].data().asnumpy()) for var in params.keys()]
        posterior_samples.attrs['epochs']=epochs
        posterior_samples.attrs['loss']=loss_val
        posterior_samples.flush()
        posterior_samples.close()
        return params,loss_val


    def predict(self,par,num_samples=100,**args):
        data_loader,n_examples=self._get_loader(**args)
        total_labels=[]
        total_samples=[]
        total_loglike=[]
        params=self.student.net.collect_params()
        for var,theta in zip(params,params.values()):
            if var in par:
                theta.data()[:]=mx.numpy.array(par[var]).copyto(self.ctx)
        for X_test,y_test in data_loader:
            X_test=X_test.as_in_context(self.ctx)
            y_test=y_test.as_in_context(self.ctx)
            y_hat=self.student.predict(par,X_test)
            total_loglike.append(y_hat.log_prob(y_test).asnumpy())
            samples=y_hat.sample_n(num_samples).asnumpy()
            total_samples.append(samples)
            total_labels.append(y_test.asnumpy())
        total_samples=np.concatenate(total_samples,axis=1)
        total_labels=np.concatenate(total_labels)
        total_loglike=np.concatenate(total_loglike)
        return total_samples,total_labels,total_loglike   