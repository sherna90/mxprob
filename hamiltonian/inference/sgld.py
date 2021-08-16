import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon,random
from mxnet.ndarray import clip
from tqdm import tqdm, trange
from copy import deepcopy
from hamiltonian.inference.base import base

class sgld(base):

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
        par=self.model.par
        for var in self.model.par.keys():
            par[var].attach_grad()
        j=0
        momentum={var:par[var].as_nd_ndarray().zeros_like(ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        samples=list()
        epsilon=self.step_size
        for i in tqdm(range(epochs)):
            data_loader,n_examples=self._get_loader(**args)
            cumulative_loss=0
            for X_batch, y_batch in data_loader:
                X_batch=X_batch.as_in_context(self.ctx)
                y_batch=y_batch.as_in_context(self.ctx)
                with autograd.record():
                    loss = self.loss(par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                epsilon=self.step_size*((30 + j) ** (-0.55))
                #epsilon=0.9*epsilon
                momentum,par=self.step(n_examples,batch_size,momentum,epsilon,self.model.par)
                cumulative_loss += nd.mean(loss).asscalar()
                j=j+1
            loss_val[i]=cumulative_loss
            file_name=chain_name+'_sgld_epoch_'+str(i)+'_.params'
            self.model.net.save_parameters(file_name)
            samples.append(file_name)
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val,samples

    def step(self,n_data,batch_size,momentum,epsilon,par):
        normal=self.draw_momentum(par,epsilon)
        for var in par.keys():
            #grad = clip(par[var].grad, -1e3,1e3)
            grad = par[var].grad/ batch_size
            grad = np.nan_to_num(grad).as_nd_ndarray()
            momentum[var][:] = self.gamma*momentum[var] + (1. - self.gamma) * nd.square(grad)
            par[var][:]=par[var]-self.step_size*grad/ nd.sqrt(momentum[var].as_nd_ndarray() + 1e-8)+normal[var].as_nd_ndarray()
            #par[var][:]=par[var]-self.step_size*grad+normal[var]


        return momentum, par

    def draw_momentum(self,par,epsilon):
        momentum={var:np.sqrt(epsilon)*random.normal(0,1,
            shape=par[var].shape,
            ctx=self.ctx,
            dtype=par[var].dtype) for var in par.keys()}
        return momentum

    def sample(self,epochs=1,batch_size=1,chains=2,verbose=False,**args):
        posterior_samples=list()
        loss_values=list()
        for i in range(chains):
            if 'chain_name' in args:
                args['chain_name']=args['chain_name']+"_"+str(i)
            _,loss,samples=self.fit(epochs=epochs,batch_size=batch_size,
                verbose=verbose,**args)
            self.model.par=self.model.reset(self.model.net)
            posterior_samples.append(samples)
            loss_values.append(loss)
        return loss_values,posterior_samples

    def predict(self,posterior_samples,num_samples,**args):
        data_loader,_=self._get_loader(**args)
        total_samples=[]
        num_samples=len(posterior_samples)
        total_loglike=[]
        for i in range(num_samples):
            samples=[]
            labels=[]
            loglike=[]
            self.model.net.load_parameters(posterior_samples[i],ctx=self.ctx)
            par=dict()
            for name,gluon_par in self.model.net.collect_params().items():
                par.update({name:gluon_par.data()})
            for X_test,y_test in data_loader:
                X_test=X_test.as_in_context(self.ctx)
                y_test=y_test.as_in_context(self.ctx)
                y_pred=self.model.predict(par,X_test)
                loglike.append(y_pred.log_prob(y_test).asnumpy())
                samples.append(y_pred.sample().asnumpy())
                labels.append(y_test.asnumpy())
            total_samples.append(np.concatenate(samples))
            total_loglike.append(np.concatenate(loglike))
            total_labels=np.concatenate(labels)
        total_samples=np.stack(total_samples)
        total_loglike=np.stack(total_loglike)
        return total_samples,total_labels,total_loglike

    def posterior_diagnostics(self,posterior_samples):
        chains=len(posterior_samples)
        posterior_samples_multiple_chains=list()
        for i in range(chains):
            single_chain={var:list() for var in self.model.par}
            for file in posterior_samples[i]:
                self.model.net.load_parameters(file,ctx=self.ctx)
                for name,par in self.model.net.collect_params().items():
                    single_chain[name].append(par.data().asnumpy())
            posterior_samples_single_chain={var:np.asarray(single_chain[var]) for var in single_chain}
            posterior_samples_multiple_chains.append(posterior_samples_single_chain)
            posterior_samples_multiple_chains_expanded=[ {var:np.expand_dims(sample,axis=0) for var,sample in posterior.items()} for posterior in posterior_samples_multiple_chains]
        return posterior_samples_multiple_chains_expanded
