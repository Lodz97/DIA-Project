import json
from environment import ClickFunction

class SystemConfiguration:
    """
    A class which set the parameters reading from configuration.json
    """

    def __init__(self):
        with open("configuration.json") as json_config:
            self.__data_config = json.load(json_config)
        json_config.close()

    def init_function(self, function_name):
        """
        :param function_name : str, name of the function selected
        :return: tuple = (bound, slope), the parameters of the function
        """
        return (self.__data_config["function_sw"][function_name]["bound"],
                self.__data_config["function_sw"][function_name]["slope"])

    def function(self,):
        func_dic = self.__data_config["function_sw"]
        tmp_list = []
        function_list = []
        for el in func_dic:
            tmp_list.append(ClickFunction.ClickFunction(func_dic[el]["bound"], func_dic[el]["slope"]))
        for i in range(0, 9, 3):
            function_list.append([tmp_list[i], tmp_list[i+1], tmp_list[i+2]])
        return function_list

    def init_sub_campaign(self, sub_campaign):
        """
        :param sub_campaign: str, name of the sub_campaign selected
        :return: dictionary, the parameters of the sub_campaign
        """
        return self.__data_config["campaign"][sub_campaign]

    def init_advertising_experiment2(self):
        return self.__data_config["Advertising_experiment2"]

    def init_advertising_experiment3(self):
        return self.__data_config["Advertising_experiment3"]

    def init_noise(self):
        return self.__data_config["campaign"]["sigma"]

    def init_learner_kernel(self):
        theta = self.__data_config["learner"]["kernel_theta"]
        l_scale = self.__data_config["learner"]["rbf _len_scale"]
        return theta, l_scale
