from learners.PricingTSLearner import PricingTSLearner
from numpy import sqrt, log


class AggregateLearner:
    """
    Represents a partition of the context generation algorithm. Each learner of this class represents a context.
    """
    def __init__(self, key_list, arms, confidence, total_aggregate):
        self.__learner = {}
        self.__translator = {}
        if not total_aggregate:
            for key in range(0, len(key_list)):
                self.__learner[key] = PricingTSLearner(len(arms), arms)
                self.__translator[key] = key_list[key]
        else:
            self.__learner[0] = PricingTSLearner(len(arms), arms)
            self.__translator[0] = key_list
        self.__confidence = confidence
        self.collected_reward = []
        self.arms = arms

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
        #print(self.__translator[self.select_learner(learner)])
        #print(self.__learner[self.select_learner(learner)].get_reward())
        self.collected_reward.append(reward*self.arms[pulled_arm])

    def number_samples(self):
        return {key[0]: key[1]._round for key in self.__learner.items()}

    def compute_lower_bound(self):
        samples_for_learner = self.number_samples()
        #print(samples_for_learner)
        lower_bound = 0
        total_n_samples = sum(samples_for_learner.values())

        for element in samples_for_learner.items():
            reward_best_arm, n_sample_arm, price = self.__learner[element[0]].get_reward_best_arm()
            bandwidth = sqrt(-log(self.__confidence)/(2*n_sample_arm))*price
            lower_bound += (reward_best_arm - bandwidth) * element[1]/total_n_samples

            #lower_bound += self.__learner[element[0]].get_reward_best_arm()[0]* element[1]/total_n_samples
            #print(element[1]/total_n_samples)
        return lower_bound

    def get_reward_best_arms(self):
        tmp = []
        for key in ["man_eu", "man_usa", "woman"]:
            tmp.append(self.__learner[self.select_learner(key)].get_reward_best_arm()[0])
        return tmp

    def print_partition_name(self):
        print(self.__translator.values())

    def get_partition_cardinality(self):
        return len(self.__learner.keys())

