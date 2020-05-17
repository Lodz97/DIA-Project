from learners import Learner
import numpy as np


class PricingGreedyLearner(Learner):
    def __init__(self, n_arms, profit_array):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)
        self.profit_array = profit_array

    def pull_arm(self):
        if self._round() < self._n_arms:
            return self._round
        idx = np.argwhere((self.profit_array * self.expected_rewards) == (self.profit_array *
                                                                          self.expected_rewards).max()).reshape(-1)
        pulled_arm = np.random.choice(idx)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        self.update_observations(pulled_arm, self.profit_array[pulled_arm] * reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self._round - 1) +
                                             self.profit_array[pulled_arm] * reward) / self._round
