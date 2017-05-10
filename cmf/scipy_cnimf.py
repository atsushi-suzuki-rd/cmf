import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from cnimf import CNIMF
import pdb

class ScipyCNIMF(CNIMF):
    def _factorize(self, n_components, convolution_width):
        return self._scipy_update(n_components, convolution_width, activation_bound = (0, None), base_bound = (0, None))


