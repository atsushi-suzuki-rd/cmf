import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from scipy.misc import logsumexp
from cnimf import CNIMF
from cnimf_lvc_nml import CNIMFLVCNML
import pdb


class VariationalCNIMF(CNIMF):
    def __init__(self,
                 convolution_max = 6, true_width = None,
                 component_max = None, true_n_components = None,
                 convergence_threshold = 0.0001, loop_max = 1000,
                 gamma_shape = 2.0, gamma_rate = 2.0,
                 base_max = 10.0):

        super().__init__(**dict(
            convolution_max = convolution_max,
            true_width = true_width,
            component_max = component_max,
            true_n_components = true_n_components,
            convergence_threshold = convergence_threshold,
            loop_max = loop_max))
        self.gamma_shape = gamma_shape
        self.gamma_rate = gamma_rate
        self.base_max = base_max

    def _prepare_special_criteria(self):
        opt_dict = dict(
            name = 'NML with LVC' ,
            convolution_max = self.convolution_max,
            component_max = self.component_max,
            gamma_shape = self.gamma_shape,
            gamma_rate = self.gamma_rate,
            base_max = self.base_max)
        self.criteria.append(CNIMFLVCNML(**opt_dict))

    def fit(self, X, y = None, filtre = None):
        X = X.astype(np.float64)
        self._fit(X, y, filtre)

    def _init_activation(self, n_components):
        return np.random.gamma(self.gamma_shape, 1.0 / self.gamma_rate, [self.n_samples, n_components])

    def _init_base(self, n_components, convolution_width):
        return np.random.uniform(0.0, self.base_max, [convolution_width, n_components, self.data_dim])

    def _update_activation(self, activation, base):
        X = self.X
        Z = activation
        Z[Z<np.finfo(float).eps] = np.finfo(float).eps
        Th = base
        Th[Th<np.finfo(float).eps] = np.finfo(float).eps
        (T, Om) = self.X.shape
        (M, K, Om) = Th.shape
        al = self.gamma_shape
        bt = self.gamma_rate
        F = self.filtre
        Lb = self.convolute(Z, Th)
        numerator = al * np.ones([T, K]) + self.reverse_convolute(F * X / Lb, Th.transpose(0,2,1)) * Z
        denominator = bt * np.ones([T, K]) + self.reverse_convolute(F, Th.transpose(0,2,1))
        newZ = numerator / denominator
        newZ[newZ<np.finfo(float).eps] = np.finfo(float).eps
        return newZ

    def _update_base(self, activation, base):
        X = self.X
        Z = activation
        Z[Z<np.finfo(float).eps] = np.finfo(float).eps
        Th = base
        Th[Th<np.finfo(float).eps] = np.finfo(float).eps
        NewTh = np.array(Th)
        (T, Om) = self.X.shape
        (M, K, Om) = Th.shape
        al = self.gamma_shape
        bt = self.gamma_rate
        F = self.filtre
        Lb = self.convolute(Z, Th)
        for m in range(M):
            numerator = (self.time_shift(Z, m).T @ (F * X / Lb)) * Th[m, :, :]
            denominator = self.time_shift(Z, m).T @ F
            NewTh[m, :, :] = numerator / denominator
        NewTh[NewTh<np.finfo(float).eps] = np.finfo(float).eps
        return NewTh
