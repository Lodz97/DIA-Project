from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import numpy as np
from sklearn import preprocessing


class EstimateHyperparameters():
    def __init__(self, function, min_x, max_x, noise_std):
        self.__function = function
        self.__domain = preprocessing.scale(np.linspace(min_x, max_x, max_x - min_x + 1))
        self.__alpha = noise_std
        self.__kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-3, 1e3))
        self.__gp = GaussianProcessRegressor(kernel=self.__kernel, alpha=self.__alpha ** 2, normalize_y=True,
                                             n_restarts_optimizer=10)

    def sample(self):
        sample = np.random.uniform(self.__domain[0], self.__domain[-1], 1000)
        target = np.random.normal(self.__function.apply_func(sample), self.__alpha)
        return sample, target

    def fit(self, sample, target):
        x = np.atleast_2d(sample).T
        y = np.array(target)
        self.__gp.fit(x, y)

    def get_parameters(self):
        return self.__gp.kernel_.get_params()
