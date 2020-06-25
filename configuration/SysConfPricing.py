import json
import numpy as np
from environment.ConversionRate import interpolate_curve, compute_aggregate_curve


class SysConfPricing:

    def __init__(self, file_path):
        self.__name = "config_pricing.json"
        with open(file_path+self.__name) as json_config:
            self.data_config = json.load(json_config)
        json_config.close()
        self._x_points = self.data_config["x_values"]
        self._prices = self.data_config["pricing_arms"]

    def get_profit(self):
        return interpolate_curve(self._x_points, self.data_config["marginal_profit"])(self._prices)

    def get_aggregate_function(self, user_prob):
        rates_tmp = [self.data_config["conversion_rate"][x] for x in ["man_eu", "man_usa", "woman"]]
        factor = self.data_config["multiplying_factor"]
        rates = [interpolate_curve(self._x_points, rates_tmp[i])(self._prices)*factor for i in range(len(rates_tmp))]
        return compute_aggregate_curve(rates, user_prob)

    def get_function(self):
        rates_tmp = [self.data_config["conversion_rate"][x] for x in ["man_eu", "man_usa", "woman"]]
        factor = self.data_config["multiplying_factor"]
        rates = [interpolate_curve(self._x_points, rates_tmp[i])(self._prices) * factor for i in range(len(rates_tmp))]
        return rates

    def get_arms_price(self):
        return self._prices

    def get_experiment_pricing_info(self):
        return self.data_config["experiment_pricing_info"]["t_horizon"], \
               self.data_config["experiment_pricing_info"]["n_experiment"]

    def get_experiment_context_info(self):
        return self.data_config["experiment_context_info"]["t_horizon"], \
               self.data_config["experiment_context_info"]["n_experiment"], \
               self.data_config["experiment_context_info"]["n_week"]

