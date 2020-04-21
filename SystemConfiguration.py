import json


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
        :return: dictionary, the parameters of the function
        """
        return self.__data_config["function"][function_name]


