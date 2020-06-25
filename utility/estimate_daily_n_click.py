from environment import BudgetEnvironment
from configuration import SysConfAdv

""" This module compute the number of click generated in a fixed number of day given a set of budgets (arms)"""

config = SysConfAdv.SysConfAdv("/home/mattia/PyProjects/DIA-Project/configuration/")
budget = config.budget_sub_campaign()
functions = config.function()
sigma = config.init_noise()
arms = [30, 40, 20]

campaign = []
for idx in range(0, len(budget)):
    campaign.append(BudgetEnvironment.BudgetEnvironment(budget[idx], sigma, functions[idx]))


def n_click_for_days(days):
    tot_number = []
    for i in range(0, days):
        tot_number.append([int(campaign[index].round(arms[index])) for index in range(0, len(campaign))])
    return tot_number


def weight(population):
    probability = []
    for en in population:
        probability.append(en / sum(population))
    return probability
