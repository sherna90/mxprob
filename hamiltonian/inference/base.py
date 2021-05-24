import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from tqdm import tqdm, trange
from copy import deepcopy
import mxnet.gluon.probability as mxp

class base:

    def __init__(self,model,start_p,step_size=0.1,ctx=mx.cpu()):
        self.start = start_p
        self.step_size = step_size
        self.model = model
        self.ctx=ctx
        self.gamma=0.9
        self.sigma=3
        self.num_batches=0
        

    def iterate_minibatches(self, X, y, batchsize):
        assert X.shape[0] == y.shape[0]
        for start_idx in range(0, X.shape[0] - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield nd.array(X[excerpt],ctx=self.ctx), nd.array(y[excerpt],ctx=self.ctx)#generador, pasa los datos originales a la memoria donde se este trabajando cpu o gpu
            #devuelve un arreglo mxnet
    def step(self,batch_size,momentum,par):
        pass

    def fit(self,epochs=1,batch_size=1,**args):
        bz = batch_size
        X=args['X_train']
        y=args['y_train']
        n_examples=X.shape[0]
        if 'verbose' in args:
            verbose=args['verbose']
        else:
            verbose=None
        epochs=int(epochs)
        loss_val=np.zeros(epochs)
        par=deepcopy(self.start)
        self.num_batches=np.ceil(y[:].shape[0]/float(batch_size))#conseguir
        decay_factor=self.step_size/self.num_batches
        initial_step_size=self.step_size
        par_momentum={var:nd.zeros_like(par[var],ctx=self.ctx) for var in par.keys()}
        std_momentum={var:nd.zeros_like(par[var],ctx=self.ctx) for var in par.keys()}
        e={var:nd.random_normal(shape=par[var].shape, loc=0., scale=1.0, ctx=self.ctx) for var in par.keys()}
        stds={var:nd.random.normal(shape=par[var].shape,ctx=self.ctx) for var in par.keys()}
        for var in par.keys():
            par[var].attach_grad()
            stds[var].attach_grad()
        #parametros para bayes by backprop
        w_0 = nd.array([1,1,1,1])
        ytensor = y.reshape([len(y),1])
        #----------------------------------
        for i in tqdm(range(epochs)):
            cumulative_loss=0
            j=0
            for X_batch, y_batch in self.iterate_minibatches(X, y,batch_size):
                with autograd.record():
                    loss = self.loss(par,X_train=X_batch,y_train=y_batch,stds=stds,e=e)
                    #loss = self.loss(par,X_train=X_batch,y_train=y_batch)
                loss.backward()#calculo de derivadas parciales de la funcion segun sus parametros. por retropropagacion
                #loss es el gradiente
                par_momentum, par = self.step(batch_size,par_momentum, par)
                std_momentum, stds = self.step(batch_size,std_momentum, stds)
                #aplicar decaimiento 
                #self.step_size = self.lr(initial_step_size,j,decay_factor,self.num_batches)
                j = j+1 
                cumulative_loss += nd.sum(loss).asscalar()
            loss_val[i]=cumulative_loss/n_examples
            #w_0 -= loss_val[i]
            if verbose and (i%(epochs/10)==0):
                print('loss: {0:.4f}'.format(loss_val[i]))
        return par,loss_val

    def lr_schedule(self,initial_step_size,step,decay_factor,num_batches):
        return initial_step_size * (1.0/(1.0+step*decay_factor*num_batches))
        #return initial_step_size - (decay_factor*step/num_batches)

    def loss(self,par,X_train,y_train):
        return self.model.loss(par,X_train=X_train,y_train=y_train)