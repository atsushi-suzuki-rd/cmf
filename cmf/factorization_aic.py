from cmf.cmf_criterion import CMFCriterion
import numpy as np

class FactorizationAIC(CMFCriterion):
    def calculate(self, divergence, activation_loss, n_samples, n_components, data_dim, convolution_width, _ = None, __ = None):
        (T, K, Om, M) = (n_samples, n_components, data_dim, convolution_width)
        return divergence + Om * K * M