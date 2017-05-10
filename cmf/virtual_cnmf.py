import sys
import numpy as np
from scipy import special
from scipy.misc import logsumexp

class VirtualCNMF:
    def __init__(self,
                 n_components = None, true_width = None, convolution_max = 6,
                 gamma_shape = 0.5, gamma_scale = 2.0,
                 convergence_threshold = 0.0001, loop_max = 1000,
                 base_max = 10.0, component_max = None):
        self.n_components = n_components
        self.true_width = true_width
        self.convolution_max = convolution_max
        self.gamma_shape = gamma_shape
        self.gamma_scale = gamma_scale
        self.loop_max = loop_max
        self.convergence_threshold = convergence_threshold
        self.base_max = base_max
        self.component_max = component_max
        self.n_methods = 5
        self.proposed = 0
        self.aic1 = 1
        self.bic1 = 2
        self.aic2 = 3
        self.bic2 = 4
        self.actvt_result = []
        self.base_result = []
        self.estimate = np.zeros([self.n_methods, 2], dtype=np.int)
        self.estimate_given_width = np.zeros(self.n_methods, dtype=np.int)
        self.best_actvt = [None for i_method in range(self.n_methods)]
        self.best_actvt_given_width = [None for i_method in range(self.n_methods)]
        self.best_base = [None for i_method in range(self.n_methods)]
        self.best_base_given_width = [None for i_method in range(self.n_methods)]
        self.best_completion = np.zeros(self.n_methods)
        self.best_completion_given_width = np.zeros(self.n_methods)

    def fit(self, X, y = None, filtre = None):
        self.X = X
        if filtre is None:
            self.filtre = np.ones(X.shape)
        else:
            self.filtre = filtre
        filtre = self.filtre
        (self.n_samples, self.data_dim) = X.shape
        if self.component_max is None:
            self.component_max = self.data_dim
        self.code_len_result = np.float("nan")\
                               * np.ones([self.convolution_max + 1,
                                          self.component_max + 1,
                                          self.loop_max])
        self.loop_cnt_result = np.float("nan")\
                        * np.ones([self.convolution_max + 1,
                                   self.component_max + 1])
        self.criterion_result = np.float("inf")\
                                * np.ones([self.n_methods,
                                           self.convolution_max + 1,
                                           self.component_max + 1])
        self.completion_result = np.float("nan")\
                               * np.ones([self.convolution_max + 1,
                                          self.component_max + 1])
        self.actvt_result = [[None for col
                              in range(self.component_max + 1)]
                             for row in range(self.convolution_max + 1)]
        self.base_result = [[None for col
                              in range(self.component_max + 1)]
                             for row in range(self.convolution_max + 1)]
        convolution_range = []
        if self.true_width is None:
            convolution_range = range(1, self.convolution_max + 1)
        else:
            convolution_range = [self.true_width]
        component_range = []
        if self.n_components is None:
            component_range = range(1, self.component_max + 1)
        else:
            component_range = [self.n_components]
        print("convolution_range", convolution_range)
        for convolution_width in convolution_range:
            log_integral_term = self._log_integral_term(convolution_width)
            print('log_integral_term', log_integral_term)
            for n_components in component_range:
                print("n_components", n_components)
                (actvt, base, code_len_transition, loop_cnt)\
                    = self._factorize(n_components, convolution_width)
                self.actvt_result[convolution_width][n_components] = actvt
                self.base_result[convolution_width][n_components] = base
                self.code_len_result[convolution_width, n_components, :]\
                    = code_len_transition
                self.loop_cnt_result[convolution_width, n_components]\
                    = loop_cnt
                self.criterion_result[:, convolution_width, n_components]\
                = self._compute_criterion(actvt, base, log_integral_term)
                self.completion_result[convolution_width, n_components]\
                = self._evaluate_completion(actvt, base)
        self._store_estimate()

    def _store_estimate(self):
        for i_method in range(0, self.n_methods):
            self.estimate[i_method, :]\
                = np.unravel_index(
                    np.nanargmin(self.criterion_result[i_method, :, :]),
                    self.criterion_result[i_method, :, :].shape)
            self.best_actvt[i_method]\
                = self.actvt_result\
                [self.estimate[i_method, 0]][self.estimate[i_method, 1]]
            self.best_base[i_method]\
                = self.base_result\
                [self.estimate[i_method, 0]][self.estimate[i_method, 1]]
            self.best_completion[i_method]\
                = self.completion_result\
                [self.estimate[i_method, 0]][self.estimate[i_method, 1]]
        if not (self.true_width is None):
            for i_method in range(0, self.n_methods):
                self.estimate_given_width[i_method]\
                    = np.nanargmin(self.criterion_result[i_method, self.true_width, :])
                self.best_actvt_given_width[i_method]\
                    = self.actvt_result\
                    [self.true_width][self.estimate_given_width[i_method]]
                self.best_base_given_width[i_method]\
                    = self.base_result\
                    [self.true_width][self.estimate_given_width[i_method]]
                self.best_completion_given_width[i_method]\
                    = self.completion_result\
                    [self.true_width][self.estimate_given_width[i_method]]

    def _factorize(self, n_components, convolution_width):
        actvt = self._init_actvt(n_components)
        base = self._init_base(n_components, convolution_width)
        new_actvt = actvt
        new_base = base
        code_len_transition = np.nan * np.ones(self.loop_max)
        loop_cnt = 0
        for loop_idx in range(0, self.loop_max):
            new_actvt = self._update_actvt(actvt, base)
            print('actvt_update', self._code_len(actvt, base))
            for i_convolution in range(0, convolution_width):
                new_base[i_convolution, :, :] = self._update_base(actvt, base, i_convolution)
            print('base_update', self._code_len(actvt, base))
            code_len_transition[loop_idx] = self._code_len(actvt, base)
            if self._is_converged(code_len_transition, loop_idx):
                loop_cnt = loop_idx
                break
            base = new_base
            actvt = new_actvt
        return (new_actvt, new_base, code_len_transition, loop_cnt)

    def _is_converged(self, code_len_transition, loop_idx):
        return  (loop_idx >= 1\
                and (code_len_transition[loop_idx - 1]\
                     - code_len_transition[loop_idx])\
                / code_len_transition[loop_idx] < self.convergence_threshold\
                and loop_idx > 0.10 * self.loop_max)\
                or (loop_idx == self.loop_max - 1)

    def _compute_criterion(self, actvt, base, log_integral_term):
        (convolution_width, n_components, data_dim) = base.shape
        criterion_value = np.float("inf") * np.ones(self.n_methods)
        code_len = self._code_len(actvt, base)
        criterion_value[self.proposed]\
            = code_len\
                  + (convolution_width * n_components * self.data_dim / 2)\
                  + np.log(self.n_samples * self.base_max * self.base_max / np.pi)\
                  + n_components * log_integral_term
        divergence = self._data_divergence(actvt, base)
        criterion_value[self.aic1]\
            = divergence\
            + convolution_width * n_components * (self.data_dim + 1)
        criterion_value[self.bic1]\
            = divergence\
            + 0.5 * convolution_width * n_components * (self.data_dim + 1)\
            * np.log(self.n_samples)
        criterion_value[self.aic2]\
            = code_len\
            + convolution_width * n_components * (self.data_dim + 1)
        criterion_value[self.bic2]\
            = code_len\
            + 0.5 * convolution_width * n_components * (self.data_dim + 1)\
            * np.log(self.n_samples)
        return criterion_value

    def _evaluate_completion(self, actvt, base):
        return self.divergence(self.X, actvt, base, np.ones(self.X.shape))

    def _data_divergence(self, actvt, base):
        return self.divergence(self.X, actvt, base, self.filtre)

    def _code_len(self, actvt, base):
        return self._data_divergence(actvt, base)\
            + self._actvt_code_len(actvt)

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
    def convolute(cls, actvt, base):
        convolution_width = base.shape[0]
        ans = actvt @ base[0,:,:]
        for i_convolution in range(1, convolution_width):
            ans += cls.time_shift(actvt, i_convolution) @ base[i_convolution, :, :]
        return ans

    @classmethod
    def reverse_convolute(cls, actvt, base):
        convolution_width = base.shape[0]
        ans = actvt @ base[0,:,:]
        for i_convolution in range(1, convolution_width):
            ans += cls.reverse_time_shift(actvt, i_convolution) @ base[i_convolution, :, :]
        return ans

    @classmethod
    def kl_divergence(cls, X, actvt, base, filtre):
        if filtre is None:
            filtre = np.ones(X.shape)
        L = cls.convolute(actvt, base)
        X = X + np.finfo(float).eps
        L = L + np.finfo(float).eps
        return (filtre * (X * (np.log(X) - np.log(L)) - X + L)).sum()

    @classmethod
    def squared_residual(cls, X, actvt, base, filtre):
        if filtre is None:
            filtre = np.ones(X.shape)
        L = cls.convolute(actvt, base)
        X = X + np.finfo(float).eps
        L = L + np.finfo(float).eps
        return (filtre * (X - L) * (X - L)).sum()

    @classmethod
    def solve_quad_eq(cls, a, half_b, c):
        ans = np.zeros(a.shape)
        d_quarter = half_b * half_b - a * c
        ans[d_quarter >= 0] = (( - half_b + np.sqrt(np.maximum(d_quarter, 0.0)))/a)[d_quarter >= 0]
        ans[d_quarter < 0 and a < 0] = float(sys.maxsize)
        return ans
