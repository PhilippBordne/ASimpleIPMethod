"""
Compared to the original implementation on 
https://github.com/osqp/osqp_benchmarks
the RandomQPExample class was modified, such that density can be passed as parameter upon construction
and that 1/4 of no. decision variables specifies no of equality constraint.
Distributed by the authors under the Apache license which can be found in the same directory. 
"""

import numpy as np
import scipy.sparse as spa
import cvxpy


class RandomQPExample(object):
    '''
    Random QP example
    '''
    def __init__(self, n, seed=1, density=0.15):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        m = int(n * 10)

        # Generate problem data
        self.n = int(n)
        self.m = m
        P = spa.random(n, n, density=density,
                       data_rvs=np.random.randn,
                       format='csc')
        self.P = P.dot(P.T).tocsc() + 1e-02 * spa.eye(n)
        self.q = np.random.randn(n)
        self.A = spa.random(m, n, density=density,
                            data_rvs=np.random.randn,
                            format='csc')
        v = np.random.randn(n)   # Fictitious solution
        
        # Philipp Bordne: here starts modification of original code
        # 1/4 of problems are candidates for equality constraint problems
        self.ne = n // 4
        self.ni = self.m - self.ne
        z = self.A@v    # point within bounds
        self.l = np.zeros(m)    # alocate memory for bounds
        self.u = np.zeros(m)
        
        # define the equality constraints
        self.l[:self.ne] = z[:self.ne]
        self.u[:self.ne] = z[:self.ne]
        
        # define the inequality constraints
        delta = np.random.rand(self.ni)  # To get inequality
        self.u[self.ne:] = z[self.ne:] + delta
        self.l[self.ne:] = - np.inf * np.ones(self.ni)  # self.u - np.random.rand(m)

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem = self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Random QP'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''
        problem = {}
        problem['P'] = self.P
        problem['q'] = self.q
        problem['A'] = self.A
        problem['l'] = self.l
        problem['u'] = self.u
        problem['m'] = self.A.shape[0]
        problem['n'] = self.A.shape[1]

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''
        x_var = cvxpy.Variable(self.n)
        objective = .5 * cvxpy.quad_form(x_var, self.P) + self.q @ x_var
        constraints = [self.A @ x_var <= self.u, self.A @ x_var >= self.l]
        problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

        return problem

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''

        variables = self.cvxpy_problem.variables()
        constraints = self.cvxpy_problem.constraints

        # primal solution
        x = variables[0].value

        # dual solution
        y = constraints[0].dual_value - constraints[1].dual_value

        return x, y
