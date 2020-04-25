import numpy as np
import itertools
import operator
from combinatorial_solver.Cell import Cell


class KnapsackSolver:

    def __init__(self, arms_dict):
        self.budgets = {}
        self.sub_campaigns_matrix = []

        self.__set_budgets(arms_dict)

    def __set_budgets(self, arms_dict):
        budget = [element for sublist in arms_dict for element in sublist.keys()]
        if 0 not in budget:
            budget.append(0)
        budget = np.unique(budget)                              # NB np.unique already returns the array sorted

        cross_prod = list(itertools.product(*[budget, budget]))

        for element in budget:
            self.budgets[element] = [comb for comb in cross_prod if np.sum(comb) == element]

    def __create_sub_campaigns_matrix(self, arms_dict):
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

        :param arms_dict: list of dictionaries.
            It contains one dictionary for each sub campaign: the keys of such dictionaries are the arms of the
            sub campaign and the values are the estimated value associated to each arm.

        :return: list of float.
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
        d = self.sub_campaigns_matrix[0]
        row = {key: Cell(d[key], np.array(key)) for key in d.keys()}

        for sb in range(1, len(self.sub_campaigns_matrix)):
            row = self.__build_table_row(self.sub_campaigns_matrix[sb], row)

        return row

    def __build_table_row(self, temp, prev_row):

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
        for key in row.keys():
            if row[key].value == -np.inf:
                row[key].alloc_array = np.zeros(0)
        return row
