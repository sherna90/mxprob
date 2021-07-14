from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import mxnet as mx

__all__ = ['FilterResponseNorm1d', 'FilterResponseNorm2d', 'FilterResponseNorm3d']


class FilterResponseNormNd(mx.gluon.HybridBlock):
    def __init__(self, num_features=0, n_dim=4, epsilon=1e-6, is_eps_learnable=False,
                 tau_initializer='zeros', beta_initializer='zeros', gamma_initializer='ones'):
        super(FilterResponseNormNd, self).__init__()
        """Filter response normalization layer (CVPR2020)
        - Arxiv: https://arxiv.org/abs/1911.09737

        Parameters
        ----------
        num_features: integer, default 0
            An integer indicating the number of dimensions of the expected input tensor.
        n_dim: integer, default 4
            An integer indicating the number of input feature dimensions.
        epsilon: float, default 1e-6
            Small float added to variance to avoid dividing by zero.
        is_eps_learnable: boolean, default False
            Indicator if to learn epsilon parameter.
        tau_initializer: str or `Initializer`, default 'zeros'
            Initializer for the tau weight.
        beta_initializer: str or `Initializer`, default 'zeros'
            Initializer for the beta weight.
        gamma_initializer: str or `Initializer`, default 'ones'
            Initializer for the gamma weight.
        """

        self.n_dim = n_dim
        self.num_features = num_features
        assert self.num_features > 0
        shape = (1, num_features) + (1,) * (n_dim - 2)

        with self.name_scope():
            self.tau = self.params.get('tau', grad_req='write',
                                       shape=shape, init=tau_initializer)
            self.gamma = self.params.get('gamma', grad_req='write',
                                         shape=shape, init=gamma_initializer)
            self.beta = self.params.get('beta', grad_req='write',
                                        shape=shape, init=beta_initializer)
            self.eps = self.params.get('eps', grad_req='write' if is_eps_learnable else 'null',
                                       shape=(1,), init=mx.initializer.Constant(epsilon))

    def cast(self, dtype):
        if np.dtype(dtype).name == 'float16':
            dtype = 'float32'
        super(FilterResponseNormNd, self).cast(dtype)

    def hybrid_forward(self, F, x, tau, gamma, beta, eps):
        avg_dims = tuple(range(2, self.n_dim))

        # Compute the mean norm of activations per channel.
        nu2 = F.mean(x ** 2, axis=avg_dims, keepdims=True)

        # Perform FilterResponseNorm.
        x = F.broadcast_mul(x, F.rsqrt(F.broadcast_add(nu2, F.abs(eps))))

        # Return after applying the Offset-ReLU non-linearity.
        return F.maximum(F.broadcast_add(F.broadcast_mul(gamma, x), beta), tau)

    def __repr__(self):
        s = '{name}({content}'
        num_features = self.num_features
        s += ', num_features={0}'.format(num_features if num_features else None)
        s += ')'
        return s.format(name=self.__class__.__name__,
                        content=', '.join(['='.join([k, v.__repr__()])
                                           for k, v in self._kwargs.items()]))


class FilterResponseNorm1d(FilterResponseNormNd):

    def __init__(self, num_features, epsilon=1e-6, is_eps_learnable=False,
                 tau_initializer='zeros', beta_initializer='zeros', gamma_initializer='ones'):
        super(FilterResponseNorm1d, self).__init__(
            num_features, n_dim=3, epsilon=epsilon, is_eps_learnable=is_eps_learnable,
            tau_initializer=tau_initializer, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer)


class FilterResponseNorm2d(FilterResponseNormNd):

    def __init__(self, num_features, epsilon=1e-6, is_eps_learnable=False,
                 tau_initializer='zeros', beta_initializer='zeros', gamma_initializer='ones'):
        super(FilterResponseNorm2d, self).__init__(
            num_features, n_dim=4, epsilon=epsilon, is_eps_learnable=is_eps_learnable,
            tau_initializer=tau_initializer, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer)


class FilterResponseNorm3d(FilterResponseNormNd):

    def __init__(self, num_features, epsilon=1e-6, is_eps_learnable=False,
                 tau_initializer='zeros', beta_initializer='zeros', gamma_initializer='ones'):
        super(FilterResponseNorm3d, self).__init__(
            num_features, n_dim=5, epsilon=epsilon, is_eps_learnable=is_eps_learnable,
            tau_initializer=tau_initializer, beta_initializer=beta_initializer, gamma_initializer=gamma_initializer)
