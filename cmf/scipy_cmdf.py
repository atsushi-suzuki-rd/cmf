import sys
import numpy as np
from numpy.linalg import solve
from scipy import special
from cmdf import CMDF
import pdb

class ScipyCMDF(CMDF):
    def _factorize(self, n_components, convolution_width):
        return self._scipy_update(n_components, convolution_width, activation_bound = (0, None), base_bound = (0, None))


