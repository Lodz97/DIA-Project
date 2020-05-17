from configuration.SystemConfiguration import SystemConfiguration
from environment import ClickFunction
import numpy as np


class SysConfAdvSW(SystemConfiguration):
    def __init__(self, path):
        self.__name = "conf_adv_sw.json"
        super(SysConfAdvSW, self).__init__(path+self.__name)

    def function(self,):
        func_dic = self.data_config["function_sw"]
        function_list = []
        for el in func_dic:
            tmp = []
            for i in range(0, 3):
                tmp.append(ClickFunction.ClickFunction(
                    func_dic[el]["bound"][i], func_dic[el]["slope"][i]))
            function_list.append(tmp)
        return function_list

    def function_name(self):
        """
        :return: The name of the function of the advertising experiment
        """
        name = []
        for key in self.data_config["function_sw"]:
            name.append(key)
        return name

    @staticmethod
    def function_list_by_phase(f_list):
        function_plot = []
        for i in range(0, 3):
            tmp = []
            tmp.append(f_list[0][i])
            tmp.append(f_list[1][i])
            tmp.append(f_list[2][i])
            function_plot.append(tmp)
        return function_plot

    def init_learner_kernel(self):
        learners_dic = self.data_config["learner"]
        param_list = []
        for el in learners_dic:
            param_list.append([np.mean(learners_dic[el]["theta"]), np.mean(learners_dic[el]["len_scale"])])
        return param_list
