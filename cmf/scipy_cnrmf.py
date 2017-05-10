import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from cnrmf import CNRMF
import pdb

class ScipyCNRMF(CNRMF):
    def _factorize(self, n_components, convolution_width):
        return self._scipy_update(n_components, convolution_width, activation_bound = (0, None), base_bound = (0, None))


