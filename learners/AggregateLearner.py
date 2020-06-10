from learners.PricingTSLearner import PricingTSLearner


class AggregateLearner:
    """
    Represents the learners of a context
    """
    def __init__(self, key_list, arms):
        self.__learner = {key: PricingTSLearner(len(arms), arms) for key in key_list}

    def select_learner(self, key_env):
        """
        Maps key of environment to key of learner contest
        string :param key_env: key of the environment
        string :return: key for the contest learner
        """
        for key in self.__learner.keys():
            if key_env in key:
                return key

    def pull_arm(self, learner):
        """
        Pull the arm of the correct learner
        string :param learner: key
        int :return: the index of pulled arm
        """
        return self.__learner[self.select_learner(learner)].pull_arm()

    def update(self, learner, pulled_arm, reward):
        """
        Update the learner after the realization
        string :param learner: learner to be updated
        int :param pulled_arm: index pulled arm
        int :param reward: realization
        """
        self.__learner[self.select_learner(learner)].update(pulled_arm, reward)

    def collected_reward(self):
        """
        float :return: the total reward
        """
        rewards = [el._collected_reward for el in self.__learner.values()]
        return sum(rewards)

    def number_samples(self):
        n_samples = [learner._round for learner in self.__learner.values()]
        return {self.__learner.keys(), n_samples}
