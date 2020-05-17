from combinatorial_solver.KnapsackSolver import KnapsackSolver
from environment.ClickFunction import ClickFunction
import numpy as np

if __name__ == "__main__":

    func = ClickFunction(350, 0.06)
    func1 = ClickFunction(100, 0.04)
    func2 = ClickFunction(100, 0.03)
    b = np.linspace(10,80,8)
    b1 = np.linspace(10,70,7)
    b2 = np.linspace(20,80,7)
    v = func.apply_func(np.linspace(10,80,8))
    v1 = func1.apply_func(np.linspace(10,70,7))
    v2 = func2.apply_func(np.linspace(20, 80, 7))
    d = {b[i]: v[i] for i in range(len(b))}
    d1 = {b1[i]: v1[i] for i in range(len(b1))}
    d2 = {b2[i]: v2[i] for i in range(len(b2))}
    l = [d, d1, d2]
    solver = KnapsackSolver(l,150)
    print(solver.solve(l))





