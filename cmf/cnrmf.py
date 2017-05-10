import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from scipy.misc import logsumexp
from cmf.cmf import CMF
import pdb

class CNRMF(CMF):
    def __init__(self,
                 convolution_max = 6, true_width = None,
                 component_max = None, true_n_components = None,
                 convergence_threshold = 0.0001, loop_max = 1000, loop_min = 0,
                 cgkm_shape = 2.0, cgkm_rate = 2.0,
                 base_max = 10.0):

        super().__init__(**dict(
            convolution_max = convolution_max,
            true_width = true_width,
            component_max = component_max,
            true_n_components = true_n_components,
            convergence_threshold = convergence_threshold,
            loop_max = loop_max,
            loop_min = loop_min))
        self.cgkm_shape = cgkm_shape
        self.cgkm_rate = cgkm_rate
        self.base_max = base_max

    def _prepare_special_criteria(self):
        pass

    def _init_activation(self, n_components):
        return np.random.gamma(self.cgkm_shape, 1.0 / self.cgkm_rate, [self.n_samples, n_components])

    def _init_base(self, n_components, convolution_width):
        return np.random.uniform(0.0, self.base_max, [convolution_width, n_components, self.data_dim])

    def fit(self, X, y = None, filtre = None, gamma_rate_vec = None):
        if gamma_rate_vec is None:
            self.gamma_rate_vec = np.ones(X.shape[1]) * 1.0
        else:
            if gamma_rate_vec.shape[0] != X.shape[1]:
                raise ValueError('gamma_rate_vec.shape[0] != X.shape[1]')
            self.gamma_rate_vec = gamma_rate_vec
        self._fit(X, y, filtre)

    def _update_activation(self, activation, base):
        X = self.X
        Z = activation
        Z[Z==0.0] = np.finfo(float).eps
        Th = base
        Th[Th==0.0] = np.finfo(float).eps
        (T, Om) = self.X.shape
        (M, K, Om) = Th.shape
        q = self.cgkm_shape
        r = self.cgkm_rate
        bt = self.gamma_rate_vec
        F = self.filtre
        Lb = self.convolute(Z, Th)
        numerator = - q * np.ones([T, K]) + self.reverse_convolute((F * np.log(X / Lb)) @ np.diag(bt), Th.transpose(0,2,1)) + np.log(Z) * self.reverse_convolute(F @ np.diag(bt), Th.transpose(0,2,1))
        denominator = r * np.ones([T, K]) + self.reverse_convolute(F @ np.diag(bt), Th.transpose(0,2,1))
        return np.exp(numerator / denominator)

    def _update_base(self, activation, base):
        X = self.X
        Z = activation
        Z[Z==0.0] = np.finfo(float).eps
        Th = base
        Th[Th==0.0] = np.finfo(float).eps
        NewTh = np.array(Th)
        (T, Om) = self.X.shape
        (M, K, Om) = Th.shape
        bt = self.gamma_rate_vec
        F = self.filtre
        Lb = self.convolute(Z, Th)
        for m in range(M):
            numerator = self.time_shift(Z, m).T @ ((F * np.log(X / Lb)) @ np.diag(bt))
            denominator = self.time_shift(Z, m).T @ (F @ np.diag(bt))
            NewTh[m, :, :] = np.exp(numerator / denominator + np.log(Th[m, :, :]))
        return NewTh

    def _activation_loss(self, activation):
        activation[activation <= 0.0] = np.finfo(float).eps
        (T, K) = activation.shape
        Z = activation
        Z[Z==0.0] = np.finfo(float).eps
        q = self.cgkm_shape
        r = self.cgkm_rate
        return (r * Z * np.log(Z) + (q - r) * Z).sum()

    def _divergence(self, X, activation, base, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        F = filtre
        L = self.convolute(activation, base)
        (T, Om) = self.X.shape
        X[X==0] = np.finfo(float).eps
        L[L==0] = np.finfo(float).eps
        bt = self.gamma_rate_vec
        try:
            (F * ((L * np.log(L/X) - L + X) @ np.diag(bt) + np.log(X))).sum()
        except RuntimeWarning:
            pdb.set_trace()
        return (F * ((L * np.log(L/X) - L + X) @ np.diag(bt) + np.log(X))).sum()
