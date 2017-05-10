import time
import numpy as np

class CallbackForScipy(object):
    def __init__(self, obj_func, maxiter):
        self.obj_func = obj_func
        self.time_origin = time.time()
        self.loss_transition = np.float("nan")  * np.ones([maxiter])
        self.elapsed_time = np.float("nan")  * np.ones([maxiter])
        self.i_iter = 0
    def __call__(self, xi):
        if self.i_iter == 0:
            self.time_origin = time.time()
        self.loss_transition[self.i_iter] = self.obj_func(xi)
        self.elapsed_time[self.i_iter] = time.time() - self.time_origin
        self.i_iter = self.i_iter + 1

