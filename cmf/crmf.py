import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from scipy.misc import logsumexp
from sklearn.decomposition import FastICA
from cmf.virtual_cmf import CMF
import pdb


class CRMF(CMF):
    def __init__(self,
                 convolution_max = None, true_width = None,
                 component_max = None, true_n_components = None,
                 convergence_threshold = 0.0001, loop_max = 1000, loop_min = 0,
                 activation_l1_weight = 2.0, activation_l2_weight = 2.0,
                 base_max = 10.0,
                 fit_accelerator_max = 0.0,
                 transfer_accelerator_max = 0.0,
                 initialization = 'smooth_svd',
                 verbose = 0):

        super().__init__(**dict(
            convolution_max = convolution_max,
            true_width = true_width,
            component_max = component_max,
            true_n_components = true_n_components,
            convergence_threshold = convergence_threshold,
            loop_max = loop_max,
            loop_min = loop_min,
            verbose = verbose,
            fit_accelerator_max = fit_accelerator_max,
            transfer_accelerator_max = transfer_accelerator_max))
        self.activation_l1_weight = np.array(activation_l1_weight)
        self.activation_l2_weight = np.array(activation_l2_weight)
        self.initialization = initialization
        self.base_max = base_max

    def fit(self, X, y = None, filtre = None, loss_weight = None, base_l1_weight = None, base_l2_weight = None):
        if loss_weight is None:
            self.loss_weight = np.mean(X * X, axis=0)
        else:
            if loss_weight.shape[0] != X.shape[1]:
                raise ValueError('loss_weight.shape[0] != X.shape[1]')
            self.loss_weight = loss_weight
        if base_l2_weight is None:
            self.base_l2_weight = np.mean(X * X, axis=0)
        else:
            if base_l2_weight.shape[0] != X.shape[1]:
                raise ValueError('base_l2_weight.shape[0] != X.shape[1]')
            self.base_l2_weight = base_l2_weight
        if base_l1_weight is None:
            self.base_l1_weight = np.zeros(X.shape[1])
        else:
            if base_l1_weight.shape[0] != X.shape[1]:
                raise ValueError('base_l1_weight.shape[0] != X.shape[1]')
            self.base_l1_weight = base_l1_weight
        self._fit(X, y, filtre)

    def _prepare_special_criteria(self):
        pass

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
            Z = np.random.uniform(-1.0, 1.0, [T, K])
            Th = np.random.uniform(-1.0, 1.0, [M, K, Om])
        return (Z, Th)

    def _init_activation_for_transfer(self, W, base, n_components, convolution_width, filtre):
        (T, Om) = W.shape
        K = n_components
        F = filtre
        Th = base
        lb = self.loss_weight
        Th0 = Th[0, :, :]
        SMALL_NUM = np.finfo(float).eps
        Z = solve(((Th0 * lb) @ Th0.T + SMALL_NUM * np.identity(K)).T, (((F * W) * lb) @ Th0.T).T).T
        return Z

    # def _init_activation(self, X, n_components):
    #     return np.random.normal(0.0, 1.0 + self.activation_l2_weight, [self.n_samples, n_components])

    # def _init_base(self, X, n_components, convolution_width):
    #     return np.random.normal(0.0, 1.0 + np.ones([convolution_width, n_components, self.data_dim]) * self.base_l2_weight[np.newaxis, np.newaxis, :], [convolution_width, n_components, self.data_dim])

    def _update_activation(self, X, activation, base, filtre, accelerator = 1.0):
        Z = activation
        Th = base
        (T, Om) = X.shape
        (M, K, Om) = Th.shape
        F = filtre
        rh = self.activation_l1_weight
        sg = self.activation_l2_weight
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
        # halfA = K * M * self.reverse_convolute(F @ np.diag(1.0 / (self.sigma * self.sigma)), (Th * Th).transpose(0, 2, 1))
        # negative_half_B = - bt * np.ones([T, K]) + self.reverse_convolute(((F * (X - Xi)) @ np.diag(1.0 / (self.sigma * self.sigma))), Th.transpose(0, 2, 1)) + halfA * Z
        # C = - (al - 1) * np.ones([T, K])
        # D_quarter = negative_half_B * negative_half_B - 2 * halfA * C
        # numerator = negative_half_B + np.sqrt(D_quarter)
        # denominator = 2 * halfA
        # return numerator / denominator

    # def _calculate_p_ut(self, Th, F, u, t):
    #     (T, Om) = self.X.shape
    #     (M, K, Om) = Th.shape
    #     P_ut_M = np.zeros([M, K, K])
    #     for m in range(max([t-u,u-t]), min([M+u-t,M+t-u,T-t])):
    #         P_ut_M[m,:,:] = Th[m+t-u, :, :] @ np.diag(F[t+m,:]) @ Th[m, :, :].T
    #     return P_ut_M.sum(axis=0)

    def _update_base(self, X, activation, base, filtre, accelerator = 1.0):
        Z = activation
        Th = base
        NewTh = np.array(Th)
        (T, Om) = X.shape
        (M, K, Om) = Th.shape
        F = filtre
        kp = self.base_l1_weight
        nu = self.base_l2_weight
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
        return NewTh

    def _activation_loss(self, activation):
        Z = activation
        return self.activation_l1_weight * np.sum(np.abs(Z)) + self.activation_l2_weight * np.sum(Z * Z)

    def _base_loss(self, base):
        Th = base
        return np.sum(np.abs(Th) * self.base_l1_weight) + np.sum(Th * Th * self.base_l2_weight)

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