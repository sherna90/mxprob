import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon,random
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base

class sgld(base):

    def fit_gluon(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain_name' in args:
            chain_name=args['chain_name']
        else:
            chain_name='chain_'+str(np.random.randint(1000)) 
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        for var in self.model.par.keys():
            self.model.par[var].attach_grad()
        sgld = mx.optimizer.Optimizer.create_optimizer('sgld',
            learning_rate=self.step_size,rescale_grad=1./batch_size)
        states=list()
        indices=list()
        samples=list()
        for i,var in enumerate(self.model.par.keys()):
            states.append(sgld.create_state(i,self.model.par[var]))
            indices.append(i)
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            j=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    loss = self.loss(self.model.par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                sgld.update(indices,[self.model.par[var] for var in self.model.par.keys()],[self.model.par[var].grad for var in self.model.par.keys()],states)
                cumulative_loss += nd.sum(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
            file_name=chain_name+'_sgld_epoch_'+str(i)+'_.params'
            self.model.net.save_parameters(file_name)
            samples.append(samples[var].append(file_name))
        return self.model.par,loss_val,samples

    def fit(self,epochs=1,batch_size=1,**args):
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        if 'chain_name' in args:
            chain_name=args['chain_name']
        else:
            chain_name='chain_'+str(np.random.randint(1000)) 
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        for var in self.model.par.keys():
            self.model.par[var].attach_grad()
        j=0
        samples=list()
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    loss = self.loss(self.model.par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                epsilon=self.step_size*((30 + j) ** (-0.55))
                momentum=self.draw_momentum(self.model.par,epsilon)
                _,self.model.par=self.step(n_examples,batch_size,momentum,epsilon,self.model.par)
                cumulative_loss += nd.sum(loss).asscalar()
                j=j+1
            loss_val[i]=cumulative_loss/n_examples
            file_name=chain_name+'_sgld_epoch_'+str(i)+'_.params'
            self.model.net.save_parameters(file_name)
            samples.append(file_name)
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return self.model.par,loss_val,samples

    def step(self,n_data,batch_size,momentum,epsilon,par):
        for name,par_new in par.items():
            momentum[name][:]=  momentum[name] - (0.5  * epsilon  * (n_data//batch_size) * par[name].grad)
            par[name][:]=par_new + momentum[name]
        return momentum, self.model.par

    def draw_momentum(self,par,epsilon):
        momentum={var:np.sqrt(epsilon)*random.normal(0,1,
            shape=par[var].shape,
            ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        return momentum

    def sample(self,epochs=1,batch_size=1,chains=2,verbose=False,**args):
        posterior_samples=list()
        for i in range(chains):
            _,_,samples=self.fit(epochs=epochs,batch_size=batch_size,**args)
            """ posterior_samples_chain=dict()
            for var in samples.keys():
                posterior_samples_chain.update(
                    {var:np.expand_dims(np.asarray(
                        [sample.asnumpy() for sample in samples[var]]),0)
                    })
            posterior_samples.append(posterior_samples_chain) """
        return posterior_samples

    def predict(self,posterior_samples,num_samples,**args):
        data_loader,_=self._get_loader(**args)
        total_samples=[]
        num_samples=len(posterior_samples)
        for i in range(num_samples):
            samples=[]
            labels=[]
            self.model.net.load_parameters(posterior_samples[i],ctx=self.ctx)
            par=dict()
            for name,gluon_par in self.model.net.collect_params().items():
                par.update({name:gluon_par.data()})
            for X_test,y_test in data_loader:
                X_test=X_test.as_in_context(self.ctx)
                y_pred=self.model.predict(par,X_test)
                samples.append(y_pred.sample().asnumpy())
                labels.append(y_test.asnumpy())
            total_samples.append(np.concatenate(samples))
            total_labels=np.concatenate(labels)
        total_samples=np.stack(total_samples)
        return total_samples,total_labels