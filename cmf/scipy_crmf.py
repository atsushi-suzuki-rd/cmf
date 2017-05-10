import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from cmf.crmf import CRMF
import pdb

class ScipyCRMF(CRMF):
    def _factorize(self, X, n_components, convolution_width, filtre):
        return self._scipy_update(X, n_components, convolution_width, filtre, activation_bound = (0, None), base_bound = (None, None))


