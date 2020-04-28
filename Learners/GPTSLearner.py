import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Learners.Learner import Learner
from sklearn import preprocessing


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
        self._means = np.zeros(n_arms)
        self._std = np.ones(n_arms)*10
        self._pulled_arms = []
        self._alpha = noise_std
        self._kernel = C(kernel_theta, (1e-3, 1e3))*RBF(len_scale, (1e-3, 1e3))
        self._gp = GaussianProcessRegressor(kernel=self._kernel, alpha=self._alpha**2, normalize_y=True,
                                            n_restarts_optimizer=9)

    def __update_observations(self, arm_idx, reward):
        """

        :param arm_idx:
        :param reward:
        :return:
        """
        self.update_observations(arm_idx, reward)
        self._pulled_arms.append(self.arms[arm_idx])

    def _update_model(self):
        """

        :return:
        """
        x_scaled = preprocessing.scale(self._pulled_arms)
        x = np.atleast_2d(x_scaled).T
        y = self._collected_rewards
        self._gp.fit(x, y)
        self.__means, self.__std = self._gp.predict(np.atleast_2d(self.arms).T, return_std=True)
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

    def pull_arm(self):
        """
        :return:
        """
        sampled_values = np.random.normal(self._means, self._std)
        sample_dic = {self.__arms[x]: sampled_values[x] for x in range(0, self._n_arms)}
        return sample_dic

    @property
    def arms(self):
        return self.__arms

    @arms.setter
    def arms(self, arms):
        self.__arms = arms
