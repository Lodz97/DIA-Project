import SystemConfiguration
import numpy as np
from environment.ClickFunction import ClickFunction

if __name__ == "__main__":

    config = SystemConfiguration.SystemConfiguration()

    param_sub_c1 = config.init_sub_campaign("sub_campaign1")
    budget_sub_c1 = np.linspace(param_sub_c1["min_budget"], param_sub_c1["max_budget"], param_sub_c1["n_arms"])
    param_sub_c2 = config.init_sub_campaign("sub_campaign2")
    budget_sub_c2 = np.linspace(param_sub_c2["min_budget"], param_sub_c2["max_budget"], param_sub_c2["n_arms"])
    param_sub_c3 = config.init_sub_campaign("sub_campaign3")
    budget_sub_c3 = np.linspace(param_sub_c3["min_budget"], param_sub_c3["max_budget"], param_sub_c3["n_arms"])

    sigma = config.init_noise()

    func_c1_p1 = ClickFunction(*config.init_function("func_man_eu_p1"))
    func_c1_p2 = ClickFunction(*config.init_function("func_man_eu_p2"))
    func_c1_p3 = ClickFunction(*config.init_function("func_man_eu_p3"))
    func_c2_p1 = ClickFunction(*config.init_function("func_man_usa_p1"))
    func_c2_p2 = ClickFunction(*config.init_function("func_man_usa_p2"))
    func_c2_p3 = ClickFunction(*config.init_function("func_man_usa_p3"))
    func_c3_p1 = ClickFunction(*config.init_function("func_woman_p1"))
    func_c3_p2 = ClickFunction(*config.init_function("func_woman_p2"))
    func_c3_p3 = ClickFunction(*config.init_function("func_woman_p3"))

