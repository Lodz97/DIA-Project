import json
import numpy as np


class SystemConfiguration:
    """
    A class which set the parameters reading from a json file
    """

    def __init__(self, path):
        with open(path) as json_config:
            self.data_config = json.load(json_config)
        json_config.close()

    def budget_sub_campaign(self):
        campaign_dic = self.data_config["campaign"]
        budgets = []
        for el in campaign_dic:
            tmp = np.linspace(campaign_dic[el]["min_budget"], campaign_dic[el]["max_budget"], campaign_dic[el]["n_arms"])
            budgets.append(tmp)
        return budgets

    def init_advertising_experiment(self):
        return self.data_config["Advertising_experiment"]

    def init_noise(self):
        return self.data_config["Advertising_experiment"]["sigma"]

