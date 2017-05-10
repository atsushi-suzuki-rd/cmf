import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from scipy.misc import logsumexp
from cmf.cmf import CMF
import pdb


class CSNMF(CMF):
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

    def fit(self, X, y = None, filtre = None, sigma = None):
        if sigma == None:
            self.sigma = np.ones(X.shape[1]) * 1.0
        else:
            if sigma.shape[0] != X.shape[1]:
                raise ValueError('sigma.shape[0] != X.shape[1]')
            self.sigma = sigma
        self._fit(X, y, filtre)

    def _prepare_special_criteria(self):
        pass

    def _init_activation(self, n_components):
        return np.random.normal(self.gamma_shape, 1.0 / self.gamma_rate, [self.n_samples, n_components])

    def _init_base(self, n_components, convolution_width):
        return np.random.uniform(-self.base_max, self.base_max, [convolution_width, n_components, self.data_dim])

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
        F = self.filtre
        Lb = self.convolute(Z, Th)
        halfA = K * M * self.reverse_convolute(F @ np.diag(1.0 / (self.sigma * self.sigma)), (Th * Th).transpose(0, 2, 1))
        negative_half_B = - bt * np.ones([T, K]) + self.reverse_convolute(((F * (X - Lb)) @ np.diag(1.0 / (self.sigma * self.sigma))), Th.transpose(0, 2, 1)) + halfA * Z
        C = - (al - 1) * np.ones([T, K])
        D_quarter = negative_half_B * negative_half_B - 2 * halfA * C
        numerator = negative_half_B + np.sqrt(D_quarter)
        denominator = 2 * halfA
        # P = np.zeros([T,T,K,K])
        # for u in range(T):
        #     for t in range(max([0, u-M]), min([T, u+M])):
        #         P[u,t,:,:] = self._calculate_p_ut(Th, F, u, t)
        # P = P.transpose([0,2,1,3]).reshape([T*K,T*K])
        # Pp = np.maximum(0, P)
        # Pn = - np.minimum(0, P)
        # Q = 2.0 * self.reverse_convolute(F * X, (Th @ np.diag(1 / (self.sigma * self.sigma))).transpose([0,2,1])).reshape([1, T*K])
        # Qp = np.maximum(0, Q)
        # Qn = - np.minimum(0, Q)
        # rowZ = np.array(Z.reshape([1, T*K]))
        # numerator = np.sqrt(2.0 * (rowZ @ Pp) + Qn + (2.0 * al - 1) / rowZ ) * rowZ
        # denominator = np.sqrt(2.0 * (rowZ @ Pn) + Qp + bt * rowZ )
        # pdb.set_trace()
        return numerator / denominator

    def _calculate_p_ut(self, Th, F, u, t):
        (T, Om) = self.X.shape
        (M, K, Om) = Th.shape
        P_ut_M = np.zeros([M, K, K])
        for m in range(max([t-u,u-t]), min([M+u-t,M+t-u,T-t])):
            P_ut_M[m,:,:] = Th[m+t-u, :, :] @ np.diag(F[t+m,:]) @ Th[m, :, :].T
        return P_ut_M.sum(axis=0)

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
        for m in range(M):
            numerator = (self.time_shift(Z, m).T @ (F * (X - Lb)))
            denominator = K*M*((self.time_shift(Z, m) * self.time_shift(Z, m)).T @ F)
            NewTh[m, :, :] = numerator / denominator + Th[m, :, :]
        return NewTh

    def _activation_loss(self, activation):
        # activation[activation <= 0.0] = np.finfo(float).eps
        # (T, K) = activation.shape
        # return - ((2 * self.gamma_shape + 1) * np.log(activation)).sum() + (self.gamma_rate * activation * activation).sum() + K * T * (- (self.gamma_shape + 1) * np.log(2.0 * self.gamma_rate) + special.gammaln(self.gamma_shape + 1))
        activation[activation <= 0.0] = np.finfo(float).eps
        (T, K) = activation.shape
        return - ((self.gamma_shape - 1) * np.log(activation)).sum() + (self.gamma_rate * activation).sum() + K * T * (- self.gamma_shape * np.log(self.gamma_rate) + special.gammaln(self.gamma_shape))

    def _divergence(self, X, activation, base, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        return self.squared_residual(X, activation, base, filtre)
