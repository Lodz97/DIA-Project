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
        self.n_sample_arm = np.zeros(len(arms))
        self.means = np.ones(len(arms))*0.5

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
        self.means[pulled_arm] = self.beta_parameters[pulled_arm, 0]/ (self.n_sample_arm[pulled_arm] + 2)
        #print(self.means*self._arms)

    def get_reward_best_arm(self):
        #beta_means = self.beta_parameters[:, 0] / (self.beta_parameters[:, 0] + self.beta_parameters[:, 1])
        exp_reward = self._arms * self.means
        #exp_reward = (exp_reward - np.mean(exp_reward))/(np.max(exp_reward) - np.min(exp_reward))
        #exp_reward = exp_reward / np.linalg.norm(exp_reward)
        idx = np.argmax(exp_reward)
        return exp_reward[idx], self.n_sample_arm[idx]+2, self._arms[idx]

    def get_reward(self):
        return self.means*self._arms

    def pull_arm_with_values(self):
        samples = self._arms * (np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])).reshape(-1)
        idx = np.argmax(samples)
        return idx, samples[idx]

    def pull_selected_arm_with_values(self, idx):
        samples = self._arms * (np.random.beta(self.beta_parameters[:, 0], self.beta_parameters[:, 1])).reshape(-1)
        return samples[idx]