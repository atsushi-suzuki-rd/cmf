from abc import *
import sys
import time
import numpy as np
from scipy import special
from scipy.misc import logsumexp
from scipy.optimize import minimize
from functools import partial
from cmf.callback_for_scipy import CallbackForScipy
from cmf.cmf_criterion import CMFCriterion
from cmf.factorization_aic import FactorizationAIC
from cmf.factorization_bic import FactorizationBIC
from cmf.lvc_aic import LVCAIC
from cmf.lvc_bic import LVCBIC
import pdb

class VirtualCMF(object, metaclass=ABCMeta):

    SIGNAL_BOUND = (None, None)
    RESPONSE_BOUND = (None, None)

    def __init__(self,
                 convolution_width,
                 n_components,
                 convergence_threshold = 0.0001, loop_max = 1000, loop_min = 0,
                 fit_accelerator_max = 0.0, transform_accelerator_max = 0.0, method = 'mu', verbose = 0):
        self.n_components = n_components
        self.convolution_width = convolution_width
        self.loop_max = loop_max
        self.loop_min = loop_min
        self.convergence_threshold = convergence_threshold
        self.fit_accelerator_max = fit_accelerator_max
        self.transform_accelerator_max = transform_accelerator_max
        self.verbose = verbose
        self.final_loop_cnt = None
        self.divergence = None
        self.signal_loss = None
        self.response_loss = None
        self.joint_loss = None
        self.completion_quality = None
        self.signal = None
        self.response = None
        self.approximate = None
        self.completion_quality = None
        self.method = method

    def fit(self, X, filtre=None):
        self._fit_transform(X, filtre)

    def fit_transform(self, X, filtre=None):
        return self._fit_transform(X, filtre)

    def transform(self, X, filtre=None):
        return self._transform(X, filtre)

    def _fit_transform(self, X, filtre = None):
        self._factorize(X, filtre, mode='fit')
        return self.signal

    def _transform(self, X, filtre = None):
        self._factorize(X, filtre, mode='transform')
        return self.signal

    def _preprocess_input(self, X):
        return X

    def _check_input(self, X):
        pass

    def _factorize(self, X, filtre, mode):
        X = X.astype(np.float64)
        self.X = X
        self._check_input(X)
        X = self._preprocess_input(X)
        if filtre is None:
            self.filtre = np.ones(X.shape)
        else:
            self.filtre = filtre
        filtre = self.filtre
        (self.n_samples, self.data_dim) = X.shape
        self.joint_loss_transition = np.full((self.loop_max, 2), np.nan)
        self.elapsed_time_transition = np.full((self.loop_max, 2), np.nan)
        if self.method == 'mu':
            (signal, response, _, _, final_loop_cnt) = self._multiplicative_update(X, filtre, mode)
        elif self.method == 'bfgs':
            (signal, response, _, _, final_loop_cnt) = self._scipy_update(X, filtre, mode)
        self.signal = signal
        if mode == 'fit':
            self.response = response
        self.approximate = self.convolute(signal, response)
        self.signal_loss = self._signal_loss(signal)
        self.response_loss = self._response_loss(response)
        self.divergence = self._divergence(X, signal, response, filtre)
        self.joint_loss = self.divergence + self.signal_loss + self.response_loss
        self.completion_quality = self._evaluate_completion(X, signal, response)
        self.final_loop_cnt = final_loop_cnt
        if self.verbose >= 1:
            print('n_components', self.n_components, 'convolution_width', self.convolution_width, 'divergence', self.divergence,
                  'joint_loss', self.joint_loss)

    def _multiplicative_update(self, X, filtre, mode):
        if mode == 'fit':
            (signal, response) = self._init_signal_response(X, filtre)
            accelerator_max = self.fit_accelerator_max
        elif mode == 'transform':
            response = self.response
            signal = self._init_signal_for_transform(X, response, filtre)
            accelerator_max = self.transform_accelerator_max
        else:
            pdb.set_trace()
        final_loop_cnt = None
        previous_loss = np.float("inf")
        loop_cnt = self.loop_max
        time_origin = time.time()
        accelerator = 10. ** (accelerator_max / loop_cnt * np.arange(0.0, loop_cnt)[::-1])
        for loop_idx in range(0, self.loop_max):
            new_signal = self._update_signal(X, signal, response, filtre, accelerator[loop_idx])
            present_loss = self._joint_loss(X, new_signal, response, filtre)
            self.joint_loss_transition[loop_idx, 0] = present_loss
            elapsed_time = time.time() - time_origin
            self.elapsed_time_transition[loop_idx, 0] = elapsed_time
            if self.verbose >= 2:
                print('loop_idx', loop_idx, 'accelerator', accelerator[loop_idx], 'elapsed_time', elapsed_time, 'joint_loss', present_loss)
            if mode == 'fit':
                new_response = self._update_response(X, new_signal, response, filtre, accelerator[loop_idx])
            elif mode == 'transform':
                new_response = response
            else:
                pdb.set_trace()
            present_loss = self._joint_loss(X, new_signal, new_response, filtre)
            self.joint_loss_transition[loop_idx, 1] = present_loss
            elapsed_time = time.time() - time_origin
            self.elapsed_time_transition[loop_idx, 1] = elapsed_time
            if self.verbose >= 2:
                print('loop_idx', loop_idx, 'accelerator', accelerator[loop_idx], 'elapsed_time', elapsed_time, 'joint_loss', present_loss)
            if np.isinf(present_loss):
                pdb.set_trace()
            if self._is_converged(present_loss, previous_loss, loop_idx) and loop_idx > self.loop_min:
                loop_cnt = loop_idx
                final_loop_cnt = loop_cnt
                break
            previous_loss = present_loss
            response = new_response
            signal = new_signal
        return (signal, response, self.joint_loss_transition, self.elapsed_time_transition, final_loop_cnt)

    def _scipy_update(self, X, filtre, mode=None):
        signal_bound = self.SIGNAL_BOUND
        response_bound = self.RESPONSE_BOUND
        if mode == 'fit':
            (signal, response) = self._init_signal_response(X, filtre)
            param_vec = self._param_mat2vec(signal, response)
            obj_fun = partial(self._vec_input_joint_loss, **{'X': X, 'filtre': filtre})
            callback = CallbackForScipy(obj_fun, self.loop_max)
            bounds = [signal_bound for i in range(np.prod(signal.shape))] + [response_bound for i in range(np.prod(response.shape))]
            res = minimize(obj_fun, param_vec, bounds = bounds, callback=callback, options={'maxiter': self.loop_max}, tol = self.convergence_threshold)
            (signal, response) = self._vec2param_mat(res['x'])
        elif mode == 'transform':
            response = self.response
            signal = self._init_signal_for_transform(X, response, filtre)
            signal_vec = self._signal2vec(signal)
            obj_fun = partial(self._signal_vec_input_joint_loss, **{'X': X, 'response': response, 'filtre': filtre})
            callback = CallbackForScipy(obj_fun, self.loop_max)
            bounds = [signal_bound for i in range(np.prod(signal.shape))]
            res = minimize(obj_fun, signal_vec, bounds = bounds, callback=callback, options={'maxiter': self.loop_max}, tol = self.convergence_threshold)
            signal = self._vec2signal(res['x'])
        else:
            pdb.set_trace()
        self.elapsed_time_transition[:, 0] = callback.elapsed_time
        self.elapsed_time_transition[:, 1] = callback.elapsed_time
        self.joint_loss_transition[:, 0] = callback.loss_transition
        self.joint_loss_transition[:, 1] = callback.loss_transition
        return (signal, response, self.joint_loss_transition, self.elapsed_time_transition, res['nit'])

    def _is_converged(self, present_loss, previous_loss, loop_idx):
        try:
            (previous_loss - present_loss) / np.abs(present_loss) < self.convergence_threshold
        except RuntimeWarning:
            pdb.set_trace()
        return (previous_loss - present_loss) / np.abs(present_loss) < self.convergence_threshold

    def _evaluate_completion(self, X, signal, response):
        return self._divergence(X, signal, response, np.ones(X.shape))

    @abstractmethod
    def _init_signal_response(self, X, filtre):
        raise NotImplementedError()

    @abstractmethod
    def _init_signal_for_transform(self, X, response, filtre):
        raise NotImplementedError()

    @abstractmethod
    def _update_signal(self, signal, response, filtre, accelerator):
        raise NotImplementedError()

    @abstractmethod
    def _update_response(self, signal, response, filtre, accelerator):
        raise NotImplementedError()

    @abstractmethod
    def _divergence(self, X, activation, base, filtre=None):
        raise NotImplementedError()

    @abstractmethod
    def _signal_loss(self, signal):
        raise NotImplementedError()

    @abstractmethod
    def _response_loss(self, response):
        raise NotImplementedError()

    def _joint_loss(self, X, activation, base, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        return self._divergence(X, activation, base, filtre) + self._signal_loss(activation) + self._response_loss(base)

    def _vec_input_joint_loss(self, param_vec, X, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        (signal, response) = self._vec2param_mat(param_vec)
        return self._joint_loss(X, signal, response, filtre)

    def _vec2param_mat(self, param_vec):
        (K, M) = (self.n_components, self.convolution_width)
        (T, Om) = self.X.shape
        signal = param_vec[:T*K].reshape([T,K])
        response = param_vec[T*K:].reshape([M,K,Om])
        return (signal, response)

    def _param_mat2vec(self, activation, base):
        (T, Om) = self.X.shape
        (M, K, Om) = base.shape
        param_vec = np.zeros([T*K+M*K*Om])
        param_vec[:T*K] = activation.reshape([T*K])
        param_vec[T*K:] = base.reshape([M*K*Om])
        return param_vec

    def _signal_vec_input_joint_loss(self, signal_vec, X, response, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        signal = self._vec2signal(signal_vec)
        return self._joint_loss(X, signal, response, filtre)

    def _vec2signal(self, signal_vec):
        (K, M) = (self.n_components, self.convolution_width)
        (T, Om) = self.X.shape
        signal = signal_vec.reshape([T, K])
        return signal

    def _signal2vec(self, signal):
        (K, M) = (self.n_components, self.convolution_width)
        (T, Om) = self.X.shape
        signal_vec = signal.reshape([T * K])
        return signal_vec

    @classmethod
    def time_shift(cls, mat, time):
        if time == 0:
            return mat
        else:
            return np.pad(mat, ((time,0),(0,0)), mode='constant')[:-time, :]

    @classmethod
    def reverse_time_shift(cls, mat, time):
        if time == 0:
            return mat
        else:
            return np.pad(mat, ((0,time),(0,0)), mode='constant')[time:, :]

    @classmethod
    def base_shift(cls, tensor, time):
        if time == 0:
            return tensor
        elif time > 0:
            return np.pad(tensor, ((time,0),(0,0),(0,0)), mode='constant')[:-time, :, :]
        elif time < 0:
            return np.pad(tensor, ((0,-time),(0,0),(0,0)), mode='constant')[-time:, :, :]

    @classmethod
    def convolute(cls, activation, base):
        convolution_width = base.shape[0]
        ans = activation @ base[0,:,:]
        for i_convolution in range(1, convolution_width):
            ans += cls.time_shift(activation, i_convolution) @ base[i_convolution, :, :]
        return ans

    @classmethod
    def reverse_convolute(cls, activation, base):
        convolution_width = base.shape[0]
        ans = activation @ base[0,:,:]
        for i_convolution in range(1, convolution_width):
            ans += cls.reverse_time_shift(activation, i_convolution) @ base[i_convolution, :, :]
        return ans

    @classmethod
    def kl_divergence(cls, X, activation, base, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        L = cls.convolute(activation, base)
        X[X<np.finfo(float).eps] = np.finfo(float).eps
        L[L<np.finfo(float).eps] = np.finfo(float).eps
        return (filtre * (X * (np.log(X) - np.log(L)) - X + L)).sum()

    @classmethod
    def squared_residual(cls, X, activation, base, filtre=None):
        if filtre is None:
            filtre = np.ones(X.shape)
        L = cls.convolute(activation, base)
        return (filtre * (X - L) * (X - L)).sum()

    @classmethod
    def solve_quad_eq(cls, a, half_b, c):
        ans = np.zeros(a.shape)
        d_quarter = half_b * half_b - a * c
        ans[d_quarter >= 0] = (( - half_b + np.sqrt(np.maximum(d_quarter, 0.0)))/a)[d_quarter >= 0]
        ans[d_quarter < 0 and a < 0] = float(sys.maxsize)
        return ans
