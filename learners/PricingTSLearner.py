from learners.Learner import Learner
import numpy as np


class PricingTSLearner(Learner):
    """
    Attributes
    ----------
    beta_parameters : np.array
        beta distribution
    _arms: list[]
        marginal profit
    """
    def __init__(self, n_arms, arms):
        super(PricingTSLearner, self).__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self._arms = arms
        self.n_sample_arm = [0]*len(arms)

    def pull_arm(self):
        idx = np.argmax(self._arms * (np.random.beta(self.beta_parameters[:, 0],
                                                            self.beta_parameters[:, 1])).reshape(-1))
        return idx

    def update(self, pulled_arm, reward):
        self._round += 1
        self.n_sample_arm[pulled_arm] += 1
        # Real reward is profit * conversion rate
        self.update_observations(pulled_arm, self._arms[pulled_arm] * reward)
        # To update beta parameters use reward between [0, 1]
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward

    def get_reward_best_arm(self):
        beta_means = self.beta_parameters[:, 0] / (self.beta_parameters[:, 0] + self.beta_parameters[:, 1])
        idx = np.argmax(self._arms * beta_means)
        return np.max(self._arms * beta_means), self.n_sample_arm[idx]
