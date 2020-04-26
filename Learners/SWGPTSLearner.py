from Learners.GPTSLearner import *


class SWGPTSLearner(GPTSLearner):

    def __init__(self, n_arms, arms, noise_std, kernel_theta, len_scale, window_size):
        super(SWGPTSLearner, self).__init__(n_arms, arms, noise_std, kernel_theta, len_scale)
        self.window_size = window_size

    def _update_model(self):

        x = np.atleast_2d(self.__pulled_arms[-self.window_size:]).T
        y = self._collected_rewards[-self.window_size]
        self.__gp.fit(x, y)
        self.__means, self.__std = self.__gp.predict(np.atleast_2d(self.arms).T, return_std=True)
        self.__std = np.maximum(self.__std, 1e-2)
