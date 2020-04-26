import Learners.Learner as Learner
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Learners.Learner import Learner


class GPTSLearner(Learner):
    """
    It is the Thompson Sampling algorithm.
    It uses regression through Gaussian Process in learning phase.
    Attributes
    ----------
     arms : list
        it is the list of the arms of the bandit algorithm
    __means : numpy.array
        it is the array which contains the means of the distribution associated to each arm
    __std : numpy.array
       it is the array which contains the standard deviations of the distribution associated to each arm
    __pulled_arms : list
        it is a list of the arms pulled during the learning process( they are stored in chronological order)
    __alpha : float
        it is the standard deviation of the noise affecting the data
    __kernel :
        it is the kernel of the Gaussian Process
    __gp: GaussianProcessRegressor
        it is the Gaussian Process regressor itself
    """

    def __init__(self, n_arms, arms, noise_std, kernel_theta, len_scale):
        super(GPTSLearner, self).__init__(n_arms)
        self.arms = arms
        self.__means = np.zeros(n_arms)
        self.__std = np.ones(n_arms)*10
        self.__pulled_arms = []
        self.__alpha = noise_std
        self.__kernel = C(kernel_theta, (1e-3, 1e3))*RBF(len_scale, (1e-3, 1e3))
        self.__gp = GaussianProcessRegressor(kernel=self.__kernel, alpha=self.__alpha**2, normalize_y=True,
                                             n_restarts_optimizer=9)

    def __update_observations(self, arm_idx, reward):
        """

        :param arm_idx:
        :param reward:
        :return:
        """
        self.update_observations(arm_idx, reward)
        self.__pulled_arms.append(self.arms[arm_idx])

    def _update_model(self):
        """

        :return:
        """
        x = np.atleast_2d(self.__pulled_arms).T
        y = self._collected_rewards
        self.__gp.fit(x, y)
        self.__means, self.__std = self.__gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.__std = np.maximum(self.__std, 1e-2)

    def update(self, pulled_arm, reward):
        """

        :param pulled_arm:
        :param reward:
        :return:
        """
        self._round += 1
        self.__update_observations(pulled_arm, reward)
        self._update_model()

    @property
    def pull_arms(self):
        """
        :return:
        """
        sampled_values = np.random.normal(self.__means, self.__std)
        sample_dic = {self.__arms[x]: sampled_values[x] for x in range(0, self._n_arms)}
        return sample_dic

    @property
    def arms(self):
        return self.__arms

    @arms.setter
    def arms(self, arms):
        self.__arms = arms
