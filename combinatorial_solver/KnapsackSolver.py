import numpy as np
from combinatorial_solver.Cell import Cell


class KnapsackSolver:

    def __init__(self, arms_dict):
        self.budgets = []
        self.sub_campaigns_matrix = np.array([])

        self.__set_budgets(arms_dict)

    def __set_budgets(self, arms_dict):
        budget = [element for sublist in arms_dict for element in sublist.keys()]
        self.budgets = np.unique(budget)                              # NB np.unique already returns the array sorted

    def __create_sub_campaigns_matrix(self, arms_dict):
        matrix = []
        for dct in arms_dict:
            for b in self.budgets:
                if b not in dct.keys():
                    dct.update({b: -np.inf})
            matrix.append([dct[key] for key in sorted(dct.keys(), reverse=False)])

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

        matrix = self.__build_table()
        last_row = [cell.value for cell in matrix[-1]]
        idx_best = np.argmax(last_row)

        return matrix[-1][idx_best].alloc_array

    def __build_table(self):
        table = np.empty((self.sub_campaigns_matrix.shape[0] + 1, self.sub_campaigns_matrix.shape[1]), dtype=Cell)

        table[0] = [Cell(0, np.zeros(0)) for i in range(0, len(self.budgets))]
        table[1] = [Cell(value, np.array([self.budgets[idx]])) for idx, value in
                    enumerate(self.sub_campaigns_matrix[0])]

        for row in range(1, np.size(self.sub_campaigns_matrix, 0)):
            # In prev_row we have results of previous iteration
            prev_row = table[row]
            temp = self.sub_campaigns_matrix[row]
            table[row + 1] = self.__build_table_row(temp, prev_row)

        return table

    def __build_table_row(self, temp, prev_row):

        row = np.array([])

        for i in range(0, temp.size):
            if temp[i] == prev_row[i].value and temp[i] == -np.inf:
                row[i] = Cell(-np.inf, np.zeros(0))

            else:
                cell_temp = np.zeros(i+1)

                for j in range(0, i+1):
                    cell_temp[j] = temp[i - j] + prev_row[j].value

                max_index = np.argmax(cell_temp)
                alloc = np.append(prev_row[max_index].alloc_array, self.budgets[i - max_index])
                row = np.append(row, [Cell(cell_temp[max_index], alloc)])

        return row
