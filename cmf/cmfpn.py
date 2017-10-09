import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from scipy.misc import logsumexp
from sklearn.decomposition import FastICA
from cmf.virtual_cmf import VirtualCMF
import pdb


class CMFPN(VirtualCMF):

    SIGNAL_BOUND = (None, None)
    RESPONSE_BOUND = (None, None)

    def __init__(self,
                 convolution_width,
                 n_components,
                 convergence_threshold=0.0001, loop_max=1000, loop_min=0,
                 signal_l1_weight=2.0, signal_l2_weight=2.0,
                 loss_weight=None, response_l1_weight=None, response_l2_weight=None,
                 fit_accelerator_max=0.0, transform_accelerator_max=0.0,
                 initialization='smooth_svd',
                 method='mu',
                 verbose=0):

        super().__init__(**dict(
            convolution_width=convolution_width,
            n_components=n_components,
            convergence_threshold=convergence_threshold,
            loop_max=loop_max,
            loop_min=loop_min,
            verbose=verbose,
            fit_accelerator_max=fit_accelerator_max,
            transform_accelerator_max=transform_accelerator_max,
            method=method))
        self.signal_l1_weight = np.array(signal_l1_weight)
        self.signal_l2_weight = np.array(signal_l2_weight)
        self.initialization = initialization
        self.loss_weight = loss_weight
        self.response_l1_weight = response_l1_weight
        self.response_l2_weight = response_l2_weight

    def _check_input(self, X):
        if self.loss_weight is None:
            self.loss_weight = np.mean(X * X, axis=0)
        elif isinstance(self.loss_weight, (int, float)):
            self.loss_weight = np.full(X.shape[1], self.loss_weight)
        else:
            if self.loss_weight.shape[0] != X.shape[1]:
                raise ValueError('loss_weight.shape[0] != X.shape[1]')
        if self.response_l2_weight is None:
            self.response_l2_weight = np.mean(X * X, axis=0)
        elif isinstance(self.response_l2_weight, (int, float)):
            self.response_l2_weight = np.full(X.shape[1], self.response_l2_weight)
        else:
            if self.response_l2_weight.shape[0] != X.shape[1]:
                raise ValueError('base_l2_weight.shape[0] != X.shape[1]')
        if self.response_l1_weight is None:
            self.response_l1_weight = np.zeros(X.shape[1])
        elif isinstance(self.response_l1_weight, (int, float)):
            self.response_l1_weight = np.full(X.shape[1], self.response_l1_weight)
        else:
            if self.response_l1_weight.shape[0] != X.shape[1]:
                raise ValueError('base_l1_weight.shape[0] != X.shape[1]')

    def _prepare_special_criteria(self):
        pass

    def _init_signal_response(self, X, filtre):
        (T, Om) = X.shape
        F = filtre
        K = self.n_components
        M = self.convolution_width
        if self.initialization == 'impulse_svd':
            SMALL_NUM = 100 * np.finfo(float).eps
            Th = np.random.uniform(-SMALL_NUM, SMALL_NUM, [M, K, Om])
            U, s, V = np.linalg.svd(F * X, full_matrices=True)
            Z0 = U[:, :K] * s[:K][np.newaxis, :]
            Z = self.reverse_time_shift(Z0, M//2)
            # Th[M//2, :, :] = V[:K, :]
        if self.initialization == 'smooth_svd':
            Th = np.zeros([M, K, Om])
            U, s, V = np.linalg.svd(F * X, full_matrices=True)
            Z = self.reverse_time_shift(U[:, :K], M//2)
            Th0 = s[:K][:, np.newaxis] * V[:K, :]
            Th = np.ones([M, 1, 1]) * Th0
        elif self.initialization == 'impulse_ica':
            ica = FastICA(n_components=K)
            Z0 = ica.fit_transform(X)
            Z = self.reverse_time_shift(Z0, M//2)
            Th0 = ica.mixing_.T
            SMALL_NUM = 100 * np.finfo(float).eps
            Th = np.random.uniform(-SMALL_NUM, SMALL_NUM, [M, K, Om])
            # Th[M//2, :, :] = Th0
        elif self.initialization == 'smooth_ica':
            ica = FastICA(n_components=K)
            Z_raw = self.reverse_time_shift(ica.fit_transform(X), M//2)
            Z_scale = np.mean(Z_raw*Z_raw, axis = 0)
            Z = Z_raw / Z_scale[np.newaxis, :]
            Th0 = Z_scale[:, np.newaxis] * ica.mixing_.T
            Th = np.ones([M, 1, 1]) * Th0
        elif self.initialization == 'random':
            Z = np.random.uniform(-1.0, 1.0, [T, K])
            Th = np.random.uniform(-1.0, 1.0, [M, K, Om])
        return (Z, Th)

    def _init_signal_for_transform(self, X, response, filtre):
        (T, Om) = X.shape
        K = self.n_components
        F = filtre
        Th = response
        lb = self.loss_weight
        Th0 = Th[0, :, :]
        SMALL_NUM = np.finfo(float).eps
        Z = solve(((Th0 * lb) @ Th0.T + SMALL_NUM * np.identity(K)).T, (((F * X) * lb) @ Th0.T).T).T
        return Z

    def _update_signal(self, X, signal, response, filtre, accelerator = 1.0):
        Z = signal
        Th = response
        (T, Om) = X.shape
        (M, K, Om) = Th.shape
        F = filtre
        rh = self.signal_l1_weight
        sg = self.signal_l2_weight
        Rh = rh * np.ones(Z.shape)
        Sg = sg * np.ones(Z.shape)
        Xi = self.convolute(Z, Th)
        H = self.convolute(np.abs(Z), np.abs(Th))
        lb = self.loss_weight
        SMALL_NUM = 10 * np.finfo(float).eps
        FXXiLTh = self.reverse_convolute((F * (X - Xi)) * lb, Th.transpose([0,2,1]))
        FHLTh = self.reverse_convolute((F * H) * lb, np.abs(Th.transpose([0,2,1])))
        # return self.shrink(FXXiLTh + FHLTh * np.sign(Z), Rh) / (FHLTh / (np.abs(Z) + SMALL_NUM * np.ones(Z.shape)) + Sg + SMALL_NUM * np.ones(Z.shape))
        raw_numerator = FXXiLTh + FHLTh * np.sign(Z)
        raw_denominator = FHLTh + Sg * np.abs(Z)
        return np.abs(Z) * self.shrink((np.abs(raw_numerator) ** accelerator) * np.sign(raw_numerator), Rh) / (((np.abs(raw_denominator) ** accelerator) * np.sign(raw_denominator)) + SMALL_NUM * np.ones(Z.shape))

    def _update_response(self, X, signal, response, filtre, accelerator = 1.0):
        Z = signal
        Th = response
        NewTh = np.array(Th)
        (T, Om) = X.shape
        (M, K, Om) = Th.shape
        F = filtre
        kp = self.response_l1_weight
        nu = self.response_l2_weight
        Kp = np.ones([K, 1]) @ kp[np.newaxis, :]
        Nu = np.ones([K, 1]) @ nu[np.newaxis, :]
        Xi = self.convolute(Z, Th)
        H = self.convolute(np.abs(Z), np.abs(Th))
        lb = self.loss_weight
        SMALL_NUM = 10 * np.finfo(float).eps
        for m in range(M):
            ZFHL = np.abs(self.time_shift(Z, m).T) @ (F * H) * lb
            ZFXXiL = self.time_shift(Z, m).T @ (F * (X - Xi)) * lb
            raw_numerator = ZFXXiL + ZFHL * np.sign(Th[m, :, :])
            raw_denominator = ZFHL + Nu * np.abs(Th[m, :, :])
            NewTh[m, :, :] = np.abs(Th[m, :, :]) * self.shrink((np.abs(raw_numerator) ** accelerator) * np.sign(raw_numerator), Kp) / ((np.abs(raw_denominator) ** accelerator) * np.sign(raw_denominator) + SMALL_NUM * np.ones(Th[m, :, :].shape))
        NewTh[np.abs(NewTh)<=SMALL_NUM] = -NewTh[np.abs(NewTh)<=SMALL_NUM]
        return NewTh

    def _signal_loss(self, signal):
        Z = signal
        return self.signal_l1_weight * np.sum(np.abs(Z)) + self.signal_l2_weight * np.sum(Z * Z)

    def _response_loss(self, response):
        Th = response
        return np.sum(np.abs(Th) * self.response_l1_weight) + np.sum(Th * Th * self.response_l2_weight)

    def _divergence(self, X, activation, base, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        F = filtre
        Z = activation
        Th = base
        diagL = np.diag(self.loss_weight)
        Xi = self.convolute(Z, Th)
        Dev = X - Xi
        lb = self.loss_weight
        return np.sum(F * Dev * Dev * lb)

    @classmethod
    def shrink(cls, shrinkee, shrinker):
        shrinkee[shrinkee>0] = np.maximum(shrinkee[shrinkee>0] - shrinker[shrinkee>0], 0)
        shrinkee[shrinkee<0] = np.minimum(shrinkee[shrinkee<0] + shrinker[shrinkee<0], 0)
        return shrinkee