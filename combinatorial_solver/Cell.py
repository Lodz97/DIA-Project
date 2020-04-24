class Cell:
    """
    This class represents a cell of the matrix used for solving the knapsack problem
    to find the best daily budget allocation among the sub campaigns.
    """
    def __init__(self, value, alloc_array):
        self.value = value
        self.alloc_array = alloc_array