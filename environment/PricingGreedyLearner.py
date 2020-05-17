from environment.PricingLearner import *


class PricingGreedyLearner(Learner):
    def __init__(self, n_arms, profit_array):
        super().__init__(n_arms)
        self.expected_rewards = np.zeros(n_arms)
        self.profit_array = profit_array

    def pull_arm(self):
        if self.t < self.n_arms:
            return self.t
        idxs = np.argwhere((self.profit_array * self.expected_rewards) == (self.profit_array *
                                                                           self.expected_rewards).max()).reshape(-1)
        pulled_arm = np.random.choice(idxs)
        return pulled_arm

    def update(self, pulled_arm, reward):
        self.t += 1
        # Real reward is profit * conversion rate
        self.update_observations(pulled_arm, self.profit_array[pulled_arm] * reward)
        self.expected_rewards[pulled_arm] = (self.expected_rewards[pulled_arm] * (self.t - 1) +
                                             self.profit_array[pulled_arm] * reward) / self.t
