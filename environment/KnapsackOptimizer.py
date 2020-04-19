import numpy as np


class DynamicOptimizer:
    """
    This class generates optimization matrix. The matrix contains in every cell the max value obtained for a given
    budget and for a given number of subcampaings and an allocation array corresponding to budget allocation to various
    subcampaigns (first position -> first subcampaign ...)
    """
    def __init__(self, subcamp_matrix, budget_array):
        """
        :param subcamp_matrix: matrix of number of clicks (obtained from GP) for every subcampaign and for every
        possible value of budget (one row per subcampaign)
        :param budget_array: array of possible budget allocations
        """
        # Self.matrix has 1 row more than subcamp_matrix
        self.matrix = np.empty((subcamp_matrix.shape[0] + 1, subcamp_matrix.shape[1]), dtype=Cell)
        self.budget_array = budget_array
        # Initialize first row (0 subcampaigns)
        for i in range(0, budget_array.size):
            self.matrix[0][i] = Cell(0, np.zeros(0))
        # Start adding subcampaigns
        for row in range(0, np.size(subcamp_matrix, 0)):
            # In prec_row we have results of previous iteration
            prec_row = self.matrix[row]
            temp = subcamp_matrix[row]
            # For every cell of the row...
            for i in range(0, temp.size):
                # For every cell, create a temporary array from which extract max value
                cell_temp = np.zeros(i + 1)
                # ... try every possible budget combination with prec_row
                for j in range(0, i + 1):
                    cell_temp[j] = temp[i - j] + prec_row[j].value
                max_index = np.argmax(cell_temp)
                # Update self.matrix[row + 1][i] cell value with max value,
                # and to compute allocation array just append the new budget value to the previous array
                self.matrix[row + 1][i] = Cell(cell_temp[max_index],
                                     np.append(self.matrix[row][max_index].alloc_array, budget_array[i - max_index]))


class Cell:
    """
    This class represents a cell of the matrix
    """
    def __init__(self, value, alloc_array):
        self.value = value
        self.alloc_array = alloc_array


# Dataset of DIA video
subc = np.array([[-np.inf, 90, 100, 105, 110, -np.inf, -np.inf, -np.inf],
                 [0, 82, 90, 92, -np.inf, -np.inf, -np.inf, -np.inf],
                 [0, 80, 83, 85, 86, -np.inf, -np.inf, -np.inf],
                 [-np.inf, 90, 110, 115, 118, 120, -np.inf, -np.inf],
                 [-np.inf, 111, 130, 138, 142, 148, 155, -np.inf]])
bud = np.array([0, 10, 20, 30, 40, 50, 60, 70])
d = DynamicOptimizer(subc, bud)
# Table formatting
spacing = 60
for i in range(0, bud.size):
    st = "Budget: " + str(bud[i])
    print(st, " " * (spacing - len(st)), "||   ", end=" ")
print("")
for i in range(0, d.matrix.shape[0]):
    for cell in range(0, d.matrix.shape[1]):
        st = "Value: " + str(d.matrix[i][cell].value) + ", allocation: " + str(d.matrix[i][cell].alloc_array)
        print(st, " " * (spacing - len(st)), "||   ", end=" ")
    print("")