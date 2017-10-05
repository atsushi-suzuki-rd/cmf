import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from scipy.misc import logsumexp
from numpy.linalg import solve
from sklearn.decomposition import FastICA
from cmf.virtual_cmf import VirtualCMF
from cmf.cnimf_lvc_nml import CNIMFLVCNML
import pdb


class CNMF(VirtualCMF):
    def __init__(self,
                 convolution_max=None, true_width=None,
                 component_max=None, true_n_components=None,
                 convergence_threshold=0.0001, loop_max=1000, loop_min=0,
                 gamma_shape=2.0, gamma_rate=2.0,
                 base_max=10.0, initialization='smooth_svd', verbose=0):

        super().__init__(**dict(
            convolution_max=convolution_max,
            true_width=true_width,
            component_max=component_max,
            true_n_components=true_n_components,
            convergence_threshold=convergence_threshold,
            loop_max=loop_max,
            loop_min=loop_min,
            verbose=verbose))
        self.gamma_shape = gamma_shape
        self.gamma_rate = gamma_rate
        self.initialization = initialization
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

    def _update_activation(self, X, activation, base, filtre, accelerator=None):
        Z = activation
        Z[Z<np.finfo(float).eps] = np.finfo(float).eps
        Th = base
        Th[Th<np.finfo(float).eps] = np.finfo(float).eps
        (T, Om) = X.shape
        (M, K, Om) = Th.shape
        al = self.gamma_shape
        bt = self.gamma_rate
        F = filtre
        Lb = self.convolute(Z, Th)
        numerator = (al - 1) * np.ones([T, K]) + self.reverse_convolute(F * X / Lb, Th.transpose(0,2,1)) * Z
        denominator = bt * np.ones([T, K]) + self.reverse_convolute(F, Th.transpose(0,2,1))
        newZ = numerator / denominator
        newZ[newZ<np.finfo(float).eps] = np.finfo(float).eps
        return newZ

    def _update_base(self, X, activation, base, filtre, accelerator=None):
        Z = activation
        Z[Z<np.finfo(float).eps] = np.finfo(float).eps
        Th = base
        Th[Th<np.finfo(float).eps] = np.finfo(float).eps
        NewTh = np.array(Th)
        (T, Om) = X.shape
        (M, K, Om) = Th.shape
        al = self.gamma_shape
        bt = self.gamma_rate
        F = filtre
        Lb = self.convolute(Z, Th)
        for m in range(M):
            numerator = (self.time_shift(Z, m).T @ (F * X / Lb)) * Th[m, :, :]
            denominator = self.time_shift(Z, m).T @ F
            NewTh[m, :, :] = numerator / denominator
        NewTh[NewTh<np.finfo(float).eps] = np.finfo(float).eps
        return NewTh

    def _init_activation_base(self, X, n_components, convolution_width, filtre):
        (T, Om) = X.shape
        F = filtre
        K = n_components
        M = convolution_width
        if self.initialization == 'impulse_svd':
            Th = np.zeros([M, K, Om])
            U, s, V = np.linalg.svd(F * X, full_matrices=True)
            Z = U[:, :K] * s[:K][np.newaxis, :]
            Th[0, :, :] = V[:K, :]
        if self.initialization == 'smooth_svd':
            Th = np.zeros([M, K, Om])
            U, s, V = np.linalg.svd(F * X, full_matrices=True)
            Z = self.reverse_time_shift(U[:, :K], M//2)
            Th0 = s[:K][:, np.newaxis] * V[:K, :]
            Th = np.ones([M, 1, 1]) * Th0
        elif self.initialization == 'impulse_ica':
            ica = FastICA(n_components=n_components)
            Z = ica.fit_transform(X)
            Th0 = ica.mixing_.T
            SMALL_NUM = 10 * np.finfo(float).eps
            Th = np.ones([M, K, Om]) * SMALL_NUM
            Th[0, :, :] = Th0
        elif self.initialization == 'smooth_ica':
            ica = FastICA(n_components=n_components)
            Z_raw = self.reverse_time_shift(ica.fit_transform(X), M//2)
            Z_scale = np.mean(Z_raw*Z_raw, axis = 0)
            Z = Z_raw / Z_scale[np.newaxis, :]
            Th0 = Z_scale[:, np.newaxis] * ica.mixing_.T
            Th = np.ones([M, 1, 1]) * Th0
        elif self.initialization == 'random':
            Z = np.random.uniform(0.0, self.base_max, [T, K])
            Th = np.random.uniform(0.0, self.base_max, [M, K, Om])
        return (np.abs(Z), np.abs(Th))

    def _init_activation_for_transform(self, W, base, n_components, convolution_width, filtre):
        (T, Om) = W.shape
        K = n_components
        F = filtre
        Th = base
        Th0 = Th[0, :, :]
        SMALL_NUM = np.finfo(float).eps
        Z = solve(((Th0) @ Th0.T + SMALL_NUM * np.identity(K)).T, (((F * W)) @ Th0.T).T).T
        return np.abs(Z)

    def _activation_loss(self, activation):
        activation[activation <= 0.0] = np.finfo(float).eps
        (T, K) = activation.shape
        return - ((self.gamma_shape - 1) * np.log(activation)).sum() + (self.gamma_rate * activation).sum() + K * T * (- self.gamma_shape * np.log(self.gamma_rate) + special.gammaln(self.gamma_shape))

    def _base_loss(self, base):
        return 0.0

    def _divergence(self, X, activation, base, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        return self.kl_divergence(X, activation, base, filtre)
