from learners.AggregateLearner import AggregateLearner
import numpy as np


class SuperSetContext:
    """
    This class manages all the possible partitions that can be created through the two features
    """
    def __init__(self, key_matrix, arms, partition_context, confidence):
        """
        list[list[]]: param key_matrix: each element is a list of key to create the single aggregate
        list[] :param arms:
        """
        self.__partition = [AggregateLearner(key_matrix[0], arms, confidence, True)]
        self.__partition = self.__partition + [AggregateLearner(key, arms, confidence, False) for key in key_matrix[1:]]
        self.active_partition = partition_context

    @property
    def active_partition(self):
        return self.__active_partition

    @active_partition.setter
    def active_partition(self, value):
        self.__active_partition = value

    def pull_arm(self, learner):
        """
        string :param learner: type of user on the site and so learner to be used inside the active context
        int :return : index pulled arm
        """
        return self.__partition[self.active_partition].pull_arm(learner)

    def update(self, learner, pulled_arm, reward):
        """
        string :param learner: type of user on the site and so learner to be used inside each context
        int :param pulled_arm: index pulled arm
        int :param reward: realization
        """
        for aggregate_l in self.__partition:
            aggregate_l.update(learner, pulled_arm, reward)

    def collected_reward(self):
        """
        float :return: collected reward of the active context
        """
        return self.__partition[self.active_partition].collected_reward

    def select_active_partition(self):
        partition_lb = [element.compute_lower_bound() for element in self.__partition]
        print(partition_lb)
        self.active_partition = np.argmax(partition_lb)

    def print_active_partition(self):
        self.__partition[self.__active_partition].print_partition_name()
