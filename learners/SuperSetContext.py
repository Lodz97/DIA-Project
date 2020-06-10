from learners.AggregateLearner import AggregateLearner


class SuperSetContext:
    """
    This class manages all the possible context (partition) that can be created through the two features
    """
    def __init__(self, key_matrix, arms, active_context):
        """
        list[list[]]: param key_matrix: each element is a list of key to create the single aggregate
        list[] :param arms:
        """
        self.__context = [AggregateLearner(key, arms) for key in key_matrix]
        self.__active_context = active_context

    @property
    def active_context(self):
        return self.__active_context

    @active_context.setter
    def active_context(self, value):
        self.__active_context = value

    def pull_arm(self, learner):
        """
        string :param learner: type of user on the site and so learner to be used inside the active context
        int :return : index pulled arm
        """
        return self.__context[self.active_context].pull_arm(learner)

    def update(self, learner, pulled_arm, reward):
        """
        string :param learner: type of user on the site and so learner to be used inside each context
        int :param pulled_arm: index pulled arm
        int :param reward: realization
        """
        for aggregate_l in self.__context:
            aggregate_l.update(learner, pulled_arm, reward)

    def collected_reward(self):
        """
        float :return: collected reward of the active context
        """
        return self.__context[self.active_context].collected_reward()