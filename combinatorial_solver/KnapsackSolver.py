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

        self.__create_sub_campaigns_matrix(arms_dict)

        matrix = self.__build_table()



    def __build_table(self):
        table = np.empty((self.sub_campaigns_matrix.shape[0] + 1, self.sub_campaigns_matrix.shape[1]), dtype=Cell)

        table[0] = [Cell(0, np.zeros(0)) for i in range(0, len(self.budgets))]
        table[1] = [Cell(value, np.zeros(0)) for value in self.sub_campaigns_matrix[0]]

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
                cell_temp = np.zeros(i + 1)

                for j in range(0, i + 1):
                    cell_temp[j] = temp[i - j] + prev_row[j].value

                max_index = np.argmax(cell_temp)
                row[i] = Cell(cell_temp[max_index], np.append(prev_row[max_index].alloc_array,
                                                              self.budgets[i - max_index]))

        return row

