import numpy as np


class Learner:
    """The base class for each learner developed for the project.

    Attributes
    ----------

    _n_arms : str
        number of different arms the learner can pull

    _round : int
        value of the current round. It has to be incremented at each round of the time horizon chosen

    _rewards_per_arm : list
        for each arm, it stores a list of the collected reward for that specific arm

    _collected_rewards : numpy.array
        array storing all the rewards obtained during the time horizon (once for round) 


    """
    def __init__(self, n_arms):
        self._n_arms = n_arms
        self._round = 0
        self._rewards_per_arm = x = [[] for i in range(n_arms)]
        self._collected_rewards = np.array([])

    def pull_arm(self):
        pass

    def update_observations(self, pulled_arm, reward):
        self._rewards_per_arm[pulled_arm].append(reward)
        self._collected_rewards = np.append(self._collected_rewards, reward)
