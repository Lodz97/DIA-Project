from learners import Learner
import numpy as np


class PricingTSLearner(Learner):
    """
    Attributes
    ----------
    beta_parameters : np.array
        beta distribution
    profit_array : list[]
        marginal profit
    """
    def __init__(self, n_arms, profit_array):
        super().__init__(n_arms)
        self.beta_parameters = np.ones((n_arms, 2))
        self.profit_array = profit_array

    def pull_arm(self):
        idx = np.argmax(self.profit_array * (np.random.beta(self.beta_parameters[:, 0],
                                                            self.beta_parameters[:, 1])).reshape(-1))
        return idx

    def update(self, pulled_arm, reward):
        self._round += 1
        # Real reward is profit * conversion rate
        self.update_observations(pulled_arm, self.profit_array[pulled_arm] * reward)
        # To update beta parameters use reward between [0, 1]
        self.beta_parameters[pulled_arm, 0] = self.beta_parameters[pulled_arm, 0] + reward
        self.beta_parameters[pulled_arm, 1] = self.beta_parameters[pulled_arm, 1] + 1.0 - reward
