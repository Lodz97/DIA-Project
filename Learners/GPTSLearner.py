import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from Learners.Learner import Learner
from sklearn import preprocessing
import matplotlib.pyplot as plt


class GPTSLearner(Learner):
    """
    It is the Thompson Sampling algorithm.
    It uses regression through Gaussian Process in learning phase.
    Attributes
    ----------
     arms : list
        it is the list of the arms of the bandit algorithm
    _means : numpy.array
        it is the array which contains the means of the distribution associated to each arm
    _std : numpy.array
       it is the array which contains the standard deviations of the distribution associated to each arm
    _pulled_arms : list
        it is a list of the arms pulled during the learning process( they are stored in chronological order)
    _alpha : float
        it is the standard deviation of the noise affecting the data
    _kernel :
        it is the kernel of the Gaussian Process
    _gp: GaussianProcessRegressor
        it is the Gaussian Process regressor itself
    """

    def __init__(self, n_arms, arms, noise_std, kernel_theta, len_scale):
        super(GPTSLearner, self).__init__(n_arms)
        self.arms = arms
        self.scaled_arms = preprocessing.scale(arms)
        self._means = np.zeros(n_arms)
        self._std = np.ones(n_arms)*10
        self._pulled_arms = []
        self._alpha = noise_std
        self._kernel = C(kernel_theta, (1e-3, 1e3))*RBF(len_scale, (1e-3, 1e3))
        self._gp = GaussianProcessRegressor(kernel=self._kernel, alpha=self._alpha**2, normalize_y=True,
                                            n_restarts_optimizer=0, optimizer=None)
        self._noise = noise_std

    def __update_observations(self, arm_idx, reward):
        """

        :param arm_idx: int
            it is the index associated to the arm pulled in the current round
        :param reward:float
            it is the reward associated to the arm, specified through its index, pulled in the current round
        :return:
        """
        self.update_observations(arm_idx, reward)
        self._pulled_arms.append(self.scaled_arms[arm_idx])

    def _update_model(self):
        """
        Method which  updates the Gaussian process regressor after having observed new samples.
        This method is called once for each round.
        """

        x = np.atleast_2d(self._pulled_arms).T
        y = self._collected_rewards
        self._gp.fit(x, y)
        self._means, self._std = self._gp.predict(np.atleast_2d(self.scaled_arms).T, return_std=True)
        self._std = np.maximum(self._std, 1e-2)
        self._std = np.minimum(self._std, self._noise)

    def update(self, pulled_arm, reward):
        """
        Method which updates the learner at each round : it updates the observations by adding the newly observed
        samples and fits the regressor ( GP process) with the updated observations.

        :param pulled_arm: int
            it is the index of the arm pulled at the current round
        :param reward: float
            it is the reward observed for the pulled arm in the current round

        """
        self._round += 1
        self.__update_observations(pulled_arm, reward)
        self._update_model()

    def pull_arm(self):
        """
        Method which pulls all the arms to extract a sample for each one of them.
        :return: dict
            It is a dictionary which associates to each arm value (keys), the corresponding sample drawn.
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

    def plot_process(self, func, t):
        """
        Methods which plots the regression obtained at time t, together with the confidence interval at 95%.

        :param func: ClickFunction
            It is the true function which has to be estimated
        :param t: int
            It is the value of the current round t
        """
        x, y = self.observed_clicks()
        x_pred = np.atleast_2d(self.scaled_arms).T
        y_predicted, sigma = self._gp.predict(x_pred, return_std=True)

        plt.figure(t)
        plt.plot(x_pred, func.apply_func(self.arms), 'r:', label=r'$func(x)$')
        plt.plot(x.ravel(), y.ravel(), 'ro', label=u'Observed Clicks')
        plt.plot(x_pred, y_predicted, 'b-', label=u'Predicted Clicks')
        plt.fill(np.concatenate([x_pred, x_pred[::-1]]),
                 np.concatenate([y_predicted - 1.96 * sigma, (y_predicted + 1.96 * sigma)[::-1]]),
                 alpha=.5, fc='b', ec='None', label='95% conf interval')
        plt.xlabel('$x$')
        plt.ylabel('$func(x)$')
        plt.legend(loc='lower right')
        plt.show()

    def observed_clicks(self):
        x = np.atleast_2d(self._pulled_arms).T
        y = self._collected_rewards
        return x, y
