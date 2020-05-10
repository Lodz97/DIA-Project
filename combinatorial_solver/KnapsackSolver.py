import numpy as np
import itertools
import operator
from combinatorial_solver.Cell import Cell


class KnapsackSolver:
    """
    It is the class which solves the knapsack-like optimization problem in order to derive the best super arm
    that has to be pulled.

    Attributes
    ----------
    budgets: dict
        It is a dictionary whose keys are the values of the possible budgets that can be chosen in a day,
        while their associated value is a list of all the possible assignments that sum up to that budget.

    sub_campaigns_matrix : list
        It is a list of dictionaries which represents the matrix associated to the sub campaigns: each row of the matrix,
        which is represented by a dictionary, matches a sub campaign. Each dictionary contains as key all the possible
        budgets that can be chosen daily and the corresponding values are the number of clicks/impressions associated to
        each value. If a budget is not defined in a sub campaign, then the corresponding value is -infinite.
    """

    def __init__(self, arms_dict, cumulative_budget):
        """
        :param arms_dict: list
            It is a list of dictionary: one dictionary for each sub campaign. The keys are the budgets associated to a
            given sub campaign. Values matching the keys are disregarded in the creation of the instance.
        """
        self.budgets = {}
        self.sub_campaigns_matrix = []
        self.__cumulative_budget = cumulative_budget
        self.__set_budgets(arms_dict)

    def __set_budgets(self, arms_dict):
        """
        Method which collects all the daily budgets of each sub campaign, adds the zero value if not already present
        (done for algorithmic fluency), and associates them with the possible combinations of existing daily budgets
        that sum up to that budget.

        :param arms_dict: list
            It contains one dictionary for each sub campaign: the keys of such dictionaries are the arms of the
            sub campaign and the values are the estimated value associated to each arm.
        """
        budget = [element for sublist in arms_dict for element in sublist.keys()]
        if 0 not in budget:
            budget.append(0)
        budget.append(self.__cumulative_budget)
        budget = np.unique(budget)                              # NB np.unique already returns the array sorted

        cross_prod = list(itertools.product(*[budget, budget]))

        for element in budget:
            self.budgets[element] = [comb for comb in cross_prod if np.sum(comb) == element]

    def __create_sub_campaigns_matrix(self, arms_dict):
        """
            Method which builds the sub campaigns matrix.
        :param arms_dict: list
            It contains one dictionary for each sub campaign: the keys of such dictionaries are the arms of the
            sub campaign and the values are the estimated value associated to each arm.
        """
        matrix = []
        for dct in arms_dict:
            for b in self.budgets.keys():
                if b not in dct.keys():
                    dct.update({b: -np.inf})
            matrix.append(dct)

        self.sub_campaigns_matrix = np.array(matrix)

    def solve(self, arms_dict):
        """
        Method called to solve the combinatorial part of the learning phase.
        It returns the best super arms found for the current round.

        :param arms_dict: list
            It contains one dictionary for each sub campaign: the keys of such dictionaries are the arms of the
            sub campaign and the values are the estimated value associated to each arm.

        :return: list
            It returns a list which represents the best super arm found, which is the solution of the knapsack-like
            problem. The list contains one arm for each sub campaign and the order matches the one which was given
            in the arms_dict parameter.
        """

        self.__create_sub_campaigns_matrix(arms_dict)

        row = self.__build_table()
        n_click = {key: row[key].value for key in row.keys()}
        max_click_key = max(n_click.items(), key=operator.itemgetter(1))[0]

        return row[max_click_key].alloc_array, row[max_click_key].value

    def __build_table(self):
        """
            Method which builds the table used in the algorithm to solve the knapsack-like problem.
            The table is not built entirely, only the last row which is the most updated one.
        :return: list
            It returns a list of dictionaries which represents the very last row of the table, the one containing the
            best allocation.
        """
        d = self.sub_campaigns_matrix[0]
        row = {key: Cell(d[key], np.array(key)) for key in d.keys()}

        for sb in range(1, len(self.sub_campaigns_matrix)):
            row = self.__build_table_row(self.sub_campaigns_matrix[sb], row)

        return row

    def __build_table_row(self, temp, prev_row):
        """
        Method which builds the row of the table.
        :param temp: list
            It is a list of dictionaries which represents the sub campaign that is currently being considered in the
            iteration of the algorithm.
        :param prev_row: list
            It is a list of dictionaries which represents the most recent row of the table, that is the partial
            allocation that has been found so far given the sub campaigns considered.
        :return: list
            It returns a list of dictionaries which represents the very last row of the table, the allocation for all
            budgets considering all the sub campaigns.

        """

        row = {}

        for b in self.budgets.keys():
            if temp[b] == prev_row[b].value and temp[b] == -np.inf:
                row[b] = Cell(-np.inf, np.zeros(0))

            else:
                cell_temp = []
                comb_tmp = []

                for idx, comb in enumerate(self.budgets[b]):
                    cell_temp.append(prev_row[comb[0]].value + temp[comb[1]])
                    comb_tmp.append(comb)

                max_index = np.argmax(cell_temp)
                row[b] = Cell(cell_temp[max_index], np.append(prev_row[comb_tmp[max_index][0]].alloc_array,
                                                              [comb_tmp[max_index][1]]))
        row = self.clean_row(row)
        return row

    @staticmethod
    def clean_row(row):
        """
        Method which clean the budget allocation for unfeasible solutions.
        :param row: list
            It is the row produced by build_table_row method.
        :return: list
            The row in input with clean allocation for unfeasible solutions.
        """
        for key in row.keys():
            if row[key].value == -np.inf:
                row[key].alloc_array = np.zeros(0)
        return row
