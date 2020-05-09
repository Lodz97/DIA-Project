from combinatorial_solver.KnapsackSolver import KnapsackSolver
import  KS


if __name__ == "__main__":
    d = {30: 110.97, 40: 130.67 , 50: 145.27 , 60: 156.08, 70: 164.1, 80: 170.03}
    d1 = {10: 74.17, 20: 123.9, 30: 157.23, 40: 179.57, 50: 194.54, 60: 204.58, 70: 211.31}
    d2 = {20: 97.83, 30: 116.85, 40: 127.29 , 50: 133.02, 60: 136.17, 70: 137.9, 80: 138.84}
    l = [d, d1, d2]
    ksol = KS.KnapsackSolver(l)
    solver = KnapsackSolver(l)
    print(solver.solve(l))
    print(ksol.solve(l))





