from learners.PricingTSLearner import PricingTSLearner
from numpy import sqrt, log, cumsum


class AggregateLearner:
    """
    Represents a partition of the context generation algorithm. Each learner of this class represents a context.
    """
    def __init__(self, key_list, arms, confidence):
        self.__learner = {key: PricingTSLearner(len(arms), arms) for key in range(0, len(key_list))}
        # mapping learners to corresponding string context
        self.__translator = {key: key_list[key] for key in range(0, len(key_list))}
        self.__confidence = confidence

    def select_learner(self, key_env):
        """
        Maps key of environment to key of learner contest
        string :param key_env: key of the environment
        string :return: key for the contest learner
        """
        for key in self.__learner.keys():
            if key_env in self.__translator[key]:
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
        rewards = [el._collected_rewards for el in self.__learner.values()]
        return sum(rewards)

    def number_samples(self):
        n_samples = [learner._round for learner in self.__learner.values()]
        return {key: n_samples[key] for key in self.__learner.keys()}

    def compute_lower_bound(self):
        samples_for_learner = self.number_samples()
        lower_bound = 0
        total_n_samples = cumsum(samples_for_learner.values())
        for element in samples_for_learner.items():
            print(element)
            lower_bound += (self.__learner[element[0]].get_reward_best_arm() -
                            sqrt(-log(self.__confidence)/element[1])) * element[1]/total_n_samples
        return lower_bound



