class model:

    def forward(self,par,**args):
        pass

    def negative_log_prior(self,par,hyper,**args):
        pass

    def negative_log_likelihood(self,par,hyper,**args):
        pass

    def loss(self,par,hyper,**args):
        pass

    def grad(self,par,hyper,**args):
        pass
