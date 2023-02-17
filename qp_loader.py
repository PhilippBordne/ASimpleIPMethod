import numpy as np
from osqp_benchmarks.problem_classes.random_qp import RandomQPExample

class ConvexQP():
    def __init__(self, n: int, seed=1) -> None:
        """
        Class interfacing the OSQP benchmark suite. Loads a problem from the OSQP benchmark suit and holds the matrices
        and vectors of the QP in the format as specified for this solver.
        n: dimension of decision variable
        """
        
        self.osqp = RandomQPExample(n, seed)
        
        self.Q = self.osqp.P.toarray()
        self.c = self.osqp.q
        self.C, self.d = self.get_inequality_constraint()
        # self.d = self.osqp.u
        
    def get_get_x_sol(self):
        self.osqp.cvxpy_problem.solve()
        return self.osqp.revert_cvxpy_solution()[0]
    
    def get_inequality_constraint(self):
        C = self.osqp.A.toarray()
        d = self.osqp.u
        # remove all constraints where row of matrix C is all 0s
        non_zero_idx = []
        for i, row in enumerate(C):
            if not np.all(row==0):
                non_zero_idx += [i]
        return C[non_zero_idx], d[non_zero_idx]
