from abc import *
import numpy as np

class CMFCriterion(object, metaclass=ABCMeta):
    def __init__(self, name, convolution_max, component_max):
        self.name = name
        self.result = np.float("inf") * np.ones([convolution_max + 1, component_max + 1])
        self.best_structure = np.float("nan") * np.ones([2])

    def joint_loss(self, X, activation, base, filtre):
        self.divergence(X, activation, base, filtre) + self.activation_loss(activation)

    def store_result(self, divergence, activation_loss, n_samples, n_components, data_dim, convolution_width, activation = None, base = None):
        self.result[convolution_width, n_components] = self.calculate(divergence, activation_loss, n_samples, n_components, data_dim, convolution_width, activation, base)

    def conclude(self, activation_result, base_result, completion_result):
        self.best_structure = np.unravel_index(np.nanargmin(self.result[:, :]), self.result[:, :].shape)
        self.best_activation = activation_result[self.best_structure[0]][self.best_structure[1]]
        self.best_base = base_result[self.best_structure[0]][self.best_structure[1]]
        self.completion_error = completion_result[self.best_structure[0]][self.best_structure[1]]

    @abstractmethod
    def calculate(self, divergence, activation_loss, n_samples, n_components, data_dim, convolution_width):
        raise NotImplementedError()
