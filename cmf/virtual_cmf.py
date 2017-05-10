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

class CMF(object, metaclass=ABCMeta):
    def __init__(self,
                 convolution_max = None, true_width = None,
                 component_max = None, true_n_components = None,
                 convergence_threshold = 0.0001, loop_max = 1000, loop_min = 0,
                 fit_accelerator_max = 0.0, transfer_accelerator_max = 0.0, verbose = 0):
        self.true_n_components = true_n_components
        self.true_width = true_width
        if convolution_max is None:
            if true_width is None:
                self.convolution_max = 6
            else:
                self.convolution_max = true_width
        else:
            self.convolution_max = convolution_max
        self.loop_max = loop_max
        self.loop_min = loop_min
        self.convergence_threshold = convergence_threshold
        self.component_max = component_max
        self.fit_accelerator_max = fit_accelerator_max
        self.transfer_accelerator_max = transfer_accelerator_max
        self.verbose = verbose

    def _prepare_criteria(self):
        self.criteria = []
        self._prepare_basic_criteria()
        self._prepare_special_criteria()

    def _prepare_basic_criteria(self):
        self.criteria.append(FactorizationAIC('factorization AIC', self.convolution_max, self.component_max))
        self.criteria.append(FactorizationBIC('factorization BIC', self.convolution_max, self.component_max))
        self.criteria.append(LVCAIC('AIC with LVC', self.convolution_max, self.component_max))
        self.criteria.append(LVCBIC('BIC with LVC', self.convolution_max, self.component_max))

    @abstractmethod
    def _prepare_special_criteria(self):
        raise NotImplementedError()

    def evaluate_criteria(self, X, filtre=None):
        result = {}
        if filtre is None:
            filtre = np.zeros(X.shape, dtype=bool)
        reversed_filtre = ~np.array(filtre, dtype=bool)
        for criterion in self.criteria:
            activation = self.transfer_activation_result[criterion.best_structure[0]][criterion.best_structure[1]]
            base = self.base_result[criterion.best_structure[0]][criterion.best_structure[1]]
            result[criterion.name] = self._divergence(X, activation, base, reversed_filtre)
        return result

    def fit(self, X, y = None, filtre = None):
        self._fit(X, y, filtre)

    def _fit(self, X, y = None, filtre = None):
        self.X = X
        if filtre is None:
            self.filtre = np.ones(X.shape)
        else:
            self.filtre = filtre
        filtre = self.filtre
        (self.n_samples, self.data_dim) = X.shape
        if self.component_max is None:
            self.component_max = self.data_dim
        self._prepare_criteria()
        self.n_methods = len(self.criteria)
        self.joint_loss_transition = np.float("nan")  * np.ones([self.convolution_max + 1, self.component_max + 1, self.loop_max, 2])
        self.elapsed_time = np.float("nan")  * np.ones([self.convolution_max + 1, self.component_max + 1, self.loop_max, 2])
        self.loop_cnt_result = np.float("nan") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.divergence_result = np.float("inf") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.activation_loss_result = np.float("inf") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.base_loss_result = np.float("inf") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.joint_loss_result = np.float("inf") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.completion_result = np.float("nan") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.activation_result = [[None for col in range(self.component_max + 1)] for row in range(self.convolution_max + 1)]
        self.base_result = [[None for col in range(self.component_max + 1)] for row in range(self.convolution_max + 1)]
        self.approximation_result = [[None for col in range(self.component_max + 1)] for row in range(self.convolution_max + 1)]
        convolution_range = []
        if self.convolution_max is None:
            if self.true_width is None:
                self.convolution_max = 10
            else:
                self.convolution_max = self.true_width
        if self.true_width is None:
            convolution_range = range(1, self.convolution_max + 1)
        else:
            convolution_range = [self.true_width]
        component_range = []
        if self.true_n_components is None:
            component_range = range(1, self.component_max + 1)
        else:
            component_range = [self.true_n_components]
        # print("convolution_range", convolution_range)
        for convolution_width in convolution_range:
            for n_components in component_range:
                # print("n_components", n_components)
                (activation, base, _, _, _)\
                    = self._factorize(X, n_components, convolution_width, filtre)
                self.activation_result[convolution_width][n_components] = activation
                self.base_result[convolution_width][n_components] = base
                self.approximation_result[convolution_width][n_components] = self.convolute(activation, base)
                activation_loss = self._activation_loss(activation)
                base_loss = self._base_loss(base)
                divergence = self._divergence(X, activation, base, filtre)
                self.activation_loss_result[convolution_width][n_components] = activation_loss
                self.base_loss_result[convolution_width][n_components] = base_loss
                self.divergence_result[convolution_width][n_components] = divergence
                joint_loss = divergence + activation_loss + base_loss
                self.joint_loss_result[convolution_width][n_components] = joint_loss
                self._compute_criterion(divergence, activation_loss, convolution_width, n_components)
                self.completion_result[convolution_width, n_components]\
                = self._evaluate_completion(X, activation, base)
                if self.verbose >= 1:
                    print('n_components', n_components, 'convolution_width', convolution_width, 'divergence', divergence, 'joint_loss', joint_loss)
        self._summarize_result()

    def _summarize_result(self):
        for criterion in self.criteria:
            criterion.conclude(self.activation_result, self.base_result, self.completion_result)
        self.activation = self.criteria[-1].best_activation
        self.base = self.criteria[-1].best_base
        self.approximated = self.convolute(self.activation, self.base)

    def _factorize(self, X, n_components, convolution_width, filtre):
        if filtre is None:
            filtre = np.ones(X.shape)
        return self._multiplicative_update(X, n_components, convolution_width, filtre)

    def _multiplicative_update(self, X, n_components, convolution_width, filtre):
        (activation, base) = self._init_activation_base(X, n_components, convolution_width, filtre)
        previous_loss = np.float("inf")
        loop_cnt = self.loop_max
        time_origin = time.time()
        accelerator_max = self.fit_accelerator_max
        accelerator = 10. ** (accelerator_max / loop_cnt * np.arange(0.0, loop_cnt)[::-1])
        for loop_idx in range(0, self.loop_max):
            new_activation = self._update_activation(X, activation, base, filtre, accelerator[loop_idx])
            present_loss = self._joint_loss(X, new_activation, base, filtre)
            self.joint_loss_transition[convolution_width, n_components, loop_idx, 0] = present_loss
            elapsed_time = time.time() - time_origin
            self.elapsed_time[convolution_width, n_components, loop_idx, 0] = elapsed_time
            if self.verbose >= 2:
                print('loop_idx', loop_idx, 'accelerator', accelerator[loop_idx], 'elapsed_time', elapsed_time, 'joint_loss', present_loss)
            new_base = self._update_base(X, new_activation, base, filtre, accelerator[loop_idx])
            present_loss = self._joint_loss(X, new_activation, new_base, filtre)
            self.joint_loss_transition[convolution_width, n_components, loop_idx, 1] = present_loss
            elapsed_time = time.time() - time_origin
            self.elapsed_time[convolution_width, n_components, loop_idx, 1] = elapsed_time
            if self.verbose >= 2:
                print('loop_idx', loop_idx, 'accelerator', accelerator[loop_idx], 'elapsed_time', elapsed_time, 'joint_loss', present_loss)
            if np.isinf(present_loss):
                pdb.set_trace()
            if self._is_converged(present_loss, previous_loss, loop_idx) and loop_idx > self.loop_min:
                loop_cnt = loop_idx
                self.loop_cnt_result[convolution_width, n_components] = loop_cnt
                break
            previous_loss = present_loss
            base = new_base
            activation = new_activation
        return (activation, base, self.joint_loss_transition[convolution_width, n_components, :, :], self.elapsed_time[convolution_width, n_components, :, :], loop_cnt)

    def transfer(self, W, y = None, transfer_filtre = None):
        self._transfer(W, y, transfer_filtre)

    def _transfer(self, W, y = None, transfer_filtre = None):
        self.W = W
        if transfer_filtre is None:
            self.transfer_filtre = np.ones(W.shape)
        else:
            self.transfer_filtre = transfer_filtre
        transfer_filtre = self.transfer_filtre
        (self.n_samples, self.data_dim) = W.shape
        if self.component_max is None:
            self.component_max = self.data_dim
        self.transfer_joint_loss_transition = np.float("nan")  * np.ones([self.convolution_max + 1, self.component_max + 1, self.loop_max])
        self.transfer_elapsed_time = np.float("nan")  * np.ones([self.convolution_max + 1, self.component_max + 1, self.loop_max])
        self.transfer_loop_cnt_result = np.float("nan") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.transfer_divergence_result = np.float("inf") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.transfer_activation_loss_result = np.float("inf") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.transfer_joint_loss_result = np.float("inf") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.transfer_completion_result = np.float("nan") * np.ones([self.convolution_max + 1, self.component_max + 1])
        self.transfer_activation_result = [[None for col in range(self.component_max + 1)] for row in range(self.convolution_max + 1)]
        self.transfer_approximation_result = [[None for col in range(self.component_max + 1)] for row in range(self.convolution_max + 1)]
        convolution_range = []
        if self.true_width is None:
            convolution_range = range(1, self.convolution_max + 1)
        else:
            convolution_range = [self.true_width]
        component_range = []
        if self.true_n_components is None:
            component_range = range(1, self.component_max + 1)
        else:
            component_range = [self.true_n_components]
        for convolution_width in convolution_range:
            for n_components in component_range:
                (activation, base, _, _, _)\
                    = self._transfer_factorize(W, n_components, convolution_width, transfer_filtre)
                self.transfer_activation_result[convolution_width][n_components] = activation
                self.transfer_approximation_result[convolution_width][n_components] = self.convolute(activation, base)
                activation_loss = self._activation_loss(activation)
                base_loss = self._base_loss(base)
                divergence = self._divergence(W, activation, base, transfer_filtre)
                self.transfer_activation_loss_result[convolution_width][n_components] = activation_loss
                self.transfer_divergence_result[convolution_width][n_components] = divergence
                joint_loss = divergence + activation_loss + base_loss
                self.transfer_joint_loss_result[convolution_width][n_components] = joint_loss
                self.transfer_completion_result[convolution_width, n_components]\
                = self._evaluate_completion(W, activation, base)
                if self.verbose >= 1:
                    print('n_components', n_components, 'convolution_width', convolution_width, 'divergence', divergence, 'joint_loss', joint_loss)
        self._summarize_transfer()

    def _summarize_transfer(self):
        self.transfer_activation = self.transfer_activation_result[self.criteria[-1].best_structure[0]][self.criteria[-1].best_structure[1]]
        self.transfer_approximated = self.convolute(self.transfer_activation, self.base)

    def _transfer_factorize(self, W, n_components, convolution_width, transfer_filtre = None):
        if transfer_filtre is None:
            self.transfer_filtre = np.ones(W.shape)
        else:
            self.transfer_filtre = transfer_filtre
        transfer_filtre = self.transfer_filtre
        return self._multiplicative_transfer(W, n_components, convolution_width, transfer_filtre)

    def _multiplicative_transfer(self, W, n_components, convolution_width, transfer_filtre):
        base = self.base_result[convolution_width][n_components]
        activation = self._init_activation_for_transfer(W, base, n_components, convolution_width, transfer_filtre)
        previous_loss = np.float("inf")
        loop_cnt = self.loop_max
        time_origin = time.time()
        accelerator_max = self.transfer_accelerator_max
        accelerator = 10. ** (accelerator_max / loop_cnt * np.arange(0.0, loop_cnt)[::-1])
        for loop_idx in range(0, self.loop_max):
            new_activation = self._update_activation(W, activation, base, transfer_filtre, accelerator[loop_idx])
            present_loss = self._joint_loss(W, new_activation, base, transfer_filtre)
            self.transfer_joint_loss_transition[convolution_width, n_components, loop_idx] = present_loss
            elapsed_time = time.time() - time_origin
            self.transfer_elapsed_time[convolution_width, n_components, loop_idx] = elapsed_time
            if self.verbose >=2:
                print('loop_idx', loop_idx, 'accelerator', accelerator[loop_idx], 'elapsed_time', elapsed_time, 'joint_loss', present_loss)
            if np.isinf(present_loss):
                pdb.set_trace()
            if self._is_converged(present_loss, previous_loss, loop_idx) and loop_idx > self.loop_min:
                loop_cnt = loop_idx
                self.loop_cnt_result[convolution_width, n_components] = loop_cnt
                break
            previous_loss = present_loss
            activation = new_activation
        return (activation, base, self.transfer_joint_loss_transition[convolution_width, n_components, :], self.transfer_elapsed_time[convolution_width, n_components, :], loop_cnt)

    def _scipy_update(self, X, n_components, convolution_width, filtre, activation_bound=(None, None), base_bound=(None, None)):
        (activation, base) = self._init_activation_base(X, n_components, convolution_width, filtre)
        param_vec = self._param_mat2vec(activation, base)
        obj_fun = partial(self._vec_input_joint_loss, **{'X': X, 'n_components': n_components, 'convolution_width': convolution_width, 'filtre': filtre})
        callback = CallbackForScipy(obj_fun, self.loop_max)
        bounds = [activation_bound for i in range(np.prod(activation.shape))] + [base_bound for i in range(np.prod(base.shape))]
        res = minimize(obj_fun, param_vec, bounds = bounds, callback=callback, options={'maxiter': self.loop_max}, tol = self.convergence_threshold)
        (activation, base) = self._vec2param_mat(res['x'], n_components, convolution_width)
        self.elapsed_time[convolution_width, n_components, :, 0] = callback.elapsed_time
        self.elapsed_time[convolution_width, n_components, :, 1] = callback.elapsed_time
        self.joint_loss_transition[convolution_width, n_components, :, 0] = callback.loss_transition
        self.joint_loss_transition[convolution_width, n_components, :, 1] = callback.loss_transition
        return (activation, base, None, None, res['nit'])

    def _is_converged(self, present_loss, previous_loss, loop_idx):
        try:
            (previous_loss - present_loss) / np.abs(present_loss) < self.convergence_threshold
        except RuntimeWarning:
            pdb.set_trace()
        return  (previous_loss - present_loss) / np.abs(present_loss) < self.convergence_threshold

    def _compute_criterion(self, divergence, activation_loss, convolution_width, n_components):
        (n_samples, data_dim) = self.X.shape
        criterion_value = np.float("inf") * np.ones(self.n_methods)
        for criterion in self.criteria:
            criterion.store_result(divergence, activation_loss, n_samples, n_components, data_dim, convolution_width)

    def _evaluate_completion(self, X, activation, base):
        return self._divergence(X, activation, base, np.ones(X.shape))

    @abstractmethod
    def _init_activation_base(self, X, n_components, convolution_width, filtre):
        raise NotImplementedError()

    @abstractmethod
    def _init_activation_for_transfer(self, W, base, n_components, convolution_width, filtre):
        raise NotImplementedError()

    @abstractmethod
    def _update_activation(self, activation, base):
        raise NotImplementedError()

    @abstractmethod
    def _update_base(self, activation, base):
        raise NotImplementedError()

    @abstractmethod
    def _divergence(self, X, activation, base, filtre = None):
        raise NotImplementedError()

    @abstractmethod
    def _activation_loss(self, activation):
        raise NotImplementedError()

    @abstractmethod
    def _base_loss(self, base):
        raise NotImplementedError()

    def _joint_loss(self, X, activation, base, filtre = None):
        if filtre is None:
            filtre = np.ones(X.shape)
        return self._divergence(X, activation, base, filtre) + self._activation_loss(activation) + self._base_loss(base)

    def _vec_input_joint_loss(self, param_vec, X, n_components, convolution_width, filtre = None):
        (K,M) = (n_components, convolution_width)
        if filtre is None:
            filtre = np.ones(X.shape)
        (T, Om) = X.shape
        (activation, base) = self._vec2param_mat(param_vec, K, M)
        return self._joint_loss(X, activation, base, filtre)

    def _vec2param_mat(self, param_vec, n_components, convolution_width):
        (K,M) = (n_components, convolution_width)
        (T, Om) = self.X.shape
        activation = param_vec[:T*K].reshape([T,K])
        base = param_vec[T*K:].reshape([M,K,Om])
        return (activation, base)

    def _param_mat2vec(self, activation, base):
        (T, Om) = self.X.shape
        (M, K, Om) = base.shape
        param_vec = np.zeros([T*K+M*K*Om])
        param_vec[:T*K] = activation.reshape([T*K])
        param_vec[T*K:] = base.reshape([M*K*Om])
        return param_vec

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
