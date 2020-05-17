from configuration.SystemConfiguration import SystemConfiguration
from environment import ClickFunction


class SysConfAdv(SystemConfiguration):
    def __init__(self, path):
        self.__name = "con_adv.json"
        super(SysConfAdv, self).__init__(path+self.__name)

    def init_learner_kernel(self):
        learners_dic = self.data_config["learner"]
        learners = []
        for el in learners_dic:
            learners.append([learners_dic[el]["theta"], learners_dic[el]["len_scale"]])
        return learners

    def function(self,):
        func_dic = self.data_config["function"]
        function_list = []
        for el in func_dic:
            function_list.append(ClickFunction.ClickFunction(func_dic[el]["bound"], func_dic[el]["slope"]))
        return function_list
