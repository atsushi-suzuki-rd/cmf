import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from scipy.misc import logsumexp
from scipy.misc import factorial
from cmf.cmf import CMF
import pdb


class CMDF(CMF):
    def __init__(self,
                 convolution_max = 6, true_width = None,
                 component_max = None, true_n_components = None,
                 convergence_threshold = 0.0001, loop_max = 1000, loop_min = 0,
                 gamma_shape = 2.0, gamma_rate = 2.0,
                 base_max = 10.0):

        super().__init__(**dict(
            convolution_max = convolution_max,
            true_width = true_width,
            component_max = component_max,
            true_n_components = true_n_components,
            convergence_threshold = convergence_threshold,
            loop_max = loop_max,
            loop_min = loop_min))
        self.gamma_shape = gamma_shape
        self.gamma_rate = gamma_rate
        self.base_max = base_max

    def _prepare_special_criteria(self):
        pass

    def _init_activation(self, n_components):
        return np.random.gamma(self.gamma_shape, 1.0 / self.gamma_rate, [self.n_samples, n_components])

    def _init_base(self, n_components, convolution_width):
        return np.random.uniform(0.0, self.base_max, [convolution_width, n_components, self.data_dim])

    def _update_activation(self, activation, base):
        X = self.X
        Z = activation
        Z[Z==0.0] = np.finfo(float).eps
        Th = base
        Th[Th==0.0] = np.finfo(float).eps
        (T, Om) = self.X.shape
        (M, K, Om) = Th.shape
        al = self.gamma_shape
        bt = self.gamma_rate
        Lb = self.convolute(Z, Th)
        Lb[Lb==0.0] = np.finfo(float).eps
        numerator = (al - 1) * np.ones([T, K]) + self.reverse_convolute(X / Lb, Th.transpose(0,2,1)) * Z
        denominator = bt * np.ones([T, K]) + self.reverse_convolute(((X @ np.ones([Om, 1])) / (Lb @ np.ones([Om, 1]))), (Th @ np.ones([Om, 1])).transpose(0,2,1))
        return numerator / denominator

    def _update_base(self, activation, base):
        X = self.X
        Z = activation
        Z[Z==0.0] = np.finfo(float).eps
        Th = base
        Th[Th==0.0] = np.finfo(float).eps
        NewTh = np.array(Th)
        (T, Om) = self.X.shape
        (M, K, Om) = Th.shape
        al = self.gamma_shape
        bt = self.gamma_rate
        F = self.filtre
        Lb = self.convolute(Z, Th)
        Lb[Lb==0.0] = np.finfo(float).eps
        for m in range(M):
            numerator = (self.time_shift(Z, m).T @ (X / Lb)) * Th[m, :, :]
            denominator = (self.time_shift(Z, m).T @ ((X @ np.ones([Om, 1])) / (Lb @ np.ones([Om, 1])))) @ np.ones([1, Om])
            NewTh[m, :, :] = numerator / denominator
        return NewTh

    def _activation_loss(self, activation):
        activation[activation <= 0.0] = np.finfo(float).eps
        (T, K) = activation.shape
        return - ((self.gamma_shape - 1) * np.log(activation)).sum() + (self.gamma_rate * activation).sum() + K * T * (- self.gamma_shape * np.log(self.gamma_rate) + special.gammaln(self.gamma_shape))

    def _divergence(self, X, activation, base, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        Z = activation
        Z[Z==0.0] = np.finfo(float).eps
        Th = base
        Th[Th==0.0] = np.finfo(float).eps
        (T, Om) = self.X.shape
        L = self.convolute(Z, Th)
        return (- special.gammaln(X.sum(axis = 1) + 1) + special.gammaln(X + 1).sum(axis = 1) - (X * (np.log(L) - np.log(L @ np.ones([Om, Om]))) ).sum(axis = 1) ).sum()
