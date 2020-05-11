from Learners.GPTSLearner import *


class SWGPTSLearner(GPTSLearner):

    def __init__(self, n_arms, arms, noise_std, kernel_theta, len_scale, window_size):
        super(SWGPTSLearner, self).__init__(n_arms, arms, noise_std, kernel_theta, len_scale)
        self.window_size = window_size

    def _update_model(self):

        x = np.atleast_2d(self._pulled_arms[-self.window_size:]).T
        y = self._collected_rewards[-self.window_size:]
        self._gp.fit(x, y)
        self._means, self._std = self._gp.predict(np.atleast_2d(self.scaled_arms).T, return_std=True)
        self._std = np.maximum(self._std, 1e-2)
        self._std = np.minimum(self._std, self._noise)

    def observed_clicks(self):
        x = np.atleast_2d(self._pulled_arms[-self.window_size:]).T
        y = self._collected_rewards[-self.window_size:]
        return x, y
