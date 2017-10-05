import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from cmf.cmfpn import CMFPN
import pdb

class ScipyCMFPN(CMFPN):
    def _factorize(self, X, n_components, convolution_width, filtre):
        return self._scipy_update(X, n_components, convolution_width, filtre, signal_bound= (0, None), response_bound= (None, None))


