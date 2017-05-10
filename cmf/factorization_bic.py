from cmf.cmf_criterion import CMFCriterion
import numpy as np

class FactorizationBIC(CMFCriterion):
    def calculate(self, divergence, activation_loss, n_samples, n_components, data_dim, convolution_width, _ = None, __ = None):
        (T, K, Om, M) = (n_samples, n_components, data_dim, convolution_width)
        return divergence + (1/2) * Om * K * M * np.log(T * Om / (2 * np.pi))