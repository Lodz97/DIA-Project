import numpy as np

"""
This file defines the possible functions that map a budget value 
to the corresponding expected number of clicks
"""


def func1(x):
    return 100 * (1.0 - np.exp(-4*x + 3*x**3))

def func2(x):
    pass

def func3(x):
    pass

# waiting for our functions
