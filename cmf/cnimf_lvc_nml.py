from cmf.cmf_criterion import CMFCriterion
import numpy as np
from scipy import special

class CNIMFLVCNML(CMFCriterion):
    def __init__(self, name, convolution_max, component_max, gamma_shape, gamma_rate, base_max):
        super().__init__(**dict(
            name = name,
            component_max = component_max,
            convolution_max = convolution_max))
        self.gamma_shape = gamma_shape
        self.gamma_rate = gamma_rate
        self.base_max = base_max

    def calculate(self, divergence, activation_loss, n_samples, n_components, data_dim, convolution_width, _ = None, __ = None):
        (T, K, Om, M) = (n_samples, n_components, data_dim, convolution_width)
        (al, bt) = (self.gamma_shape, self.gamma_rate)
        return divergence + activation_loss + (1/2) * Om * K * M * np.log(T / (2 * np.pi)) + (1/2) * Om * K * M * np.log(4.0 * al * self.base_max / bt)
        # return divergence + activation_loss + (1/2) * Om * K * M * np.log(T / (2 * np.pi))\
        # + (1/2) * Om * K * np.log(4 * self.base_max) - K * special.gammaln(self.gamma_shape)\
        # + K * (1/T) * np.sum(special.gammaln(self.gamma_shape - (Om / 2) * np.minimum(M * np.ones([T]) , (T * np.ones([T]) - np.mgrid[0:T])) - (Om / 2) * (T * np.ones([T]) - np.mgrid[0:T]) * np.log(self.gamma_rate) ) )
