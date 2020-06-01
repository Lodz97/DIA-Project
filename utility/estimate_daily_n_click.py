from environment import BudgetEnvironment
from configuration import SysConfAdv

config = SysConfAdv.SysConfAdv("/home/orso/Documents/POLIMI/DataIntelligenceApplication/DIA-Project/configuration/")
budget = config.budget_sub_campaign()
functions = config.function()
sigma = config.init_noise()
arms = [30, 40, 20]

campaign = []
for idx in range(0, len(budget)):
    campaign.append(BudgetEnvironment.BudgetEnvironment(budget[idx], sigma, functions[idx]))


def n_click_for_days(days):
    tot_number = 0
    for i in range(0, days):
        for index in range(0, len(campaign)):
            tot_number += campaign[index].round(arms[index])
    return tot_number

