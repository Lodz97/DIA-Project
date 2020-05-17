from environment.AbstractClassEnvironment import AbstractClassEnvironment
import numpy as np


class PricingEnvironment(AbstractClassEnvironment):
    def __init__(self, n_arms, probabilities):
        
        AbstractClassEnvironment.__init__(self)
        self.n_arms = n_arms
        self.probabilities = probabilities

    def round(self, pulled_arm):
        reward = np.random.binomial(1, self.probabilities[pulled_arm])
        return reward
