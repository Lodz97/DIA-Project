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

    def __update_observations(self, arm_idx, reward):
        self.update_observations(arm_idx, reward)
        self.__pulled_arms.append(self.__arms[arm_idx])

    def __update_model(self):
        x = np.atleast_2d(self.__pulled_arms).T
        y = self._collected_rewards
        self.__gp.fit(x, y)
        self.__means, self.__std = self.gp.predict(np.atleast_2d(self.__arms).T, return_std=True)

    def update(self, pulled_arm, reward):
        self._round += 1
        self.__update_observations(pulled_arm, reward)
        self.__update_model()

    def pull_arms(self):
        sampled_values = np.random.normal(self.__means, self.__std)
        return sampled_values
