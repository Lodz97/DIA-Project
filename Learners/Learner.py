import numpy as np


class Learner:
    """The base class for each learner developed for the project.

    Attributes
    ----------

    __n_arms : str
        number of different arms the learner can pull

    __round : int
        value of the current round. It has to be incremented at each round of the time horizon chosen

    __rewards_per_arm : list
        for each arm, it stores a list of the collected reward for that specific arm

    __collected_rewards : numpy.array
        array storing all the rewards obtained during the time horizon (once for round) 


    """
    def __init__(self, n_arms):
        self.__n_arms = n_arms
        self.__round = 0
        self.__rewards_per_arm = x = [[] for i in range(n_arms)]
        self.__collected_rewards = np.array([])

    def pull_arm(self):
        pass

    def update_observations(self, pulled_arm, reward):
        self.__rewards_per_arm[pulled_arm].append(reward)
        self.__collected_rewards = np.append(self.__collected_rewards, reward)
