import Learners.Learner as Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


class GPTSLearner(Learner):
    def __init__(self, n_arms, arms, noise_std):
        super.__init__(n_arms)
        self.__arms = arms
        self.__means = np.zeros(n_arms)
        self.__std = np.ones(n_arms)*10
        self.__pulled_arms = []
        self.__alpha = noise_std
        self.__kernel = C(1.0, (1e-3, 1e3))*RBF(1.0, (1e-3, 1e3))       # TODO make the params be configurable
        self.__gp = GaussianProcessRegressor(kernel=self.__kernel, alpha=self.__alpha**2, normalize_y=True,
                                             n_restarts_optimizer=9)
        
