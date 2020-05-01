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

    def plot_process(self, func, t):
        x = np.atleast_2d(self._pulled_arms[-self.window_size:]).T
        x_pred = np.atleast_2d(self.scaled_arms).T
        y = self._collected_rewards[-self.window_size:]
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