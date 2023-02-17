import numpy as np
from osqp_benchmarks.problem_classes.random_qp import RandomQPExample
from ldlt_solver import LinSysSolver

class ConvexQP():
    def __init__(self, n: int, seed=1, store_traj=False, max_iter=100) -> None:
        """
        Class interfacing the OSQP benchmark suite. Loads a problem from the OSQP benchmark suit and holds the matrices
        and vectors of the QP in the format as specified for this solver.
        n: dimension of decision variable
        """
        
        self.osqp = RandomQPExample(n, seed)
        
        self.Q = self.osqp.P.toarray()
        self.c = self.osqp.q
        self.A, self.b = self.get_equality_constraint()
        self.C, self.d = self.get_inequality_constraint()
        
        self.nx = len(self.Q)
        self.ne = len(self.A)
        self.ni = len(self.C)
        
        self.idx_x = 0
        self.idx_l = self.nx
        self.idx_mu = self.nx + self.ne
        self.idx_s = self.nx + self.ne + self.ni
        
        self.x = self.get_x_init()
        self.l = self.get_l_init()
        self.mu = self.get_mu_init()
        self.s = self.get_s_init()
        
        self.store_traj = store_traj
        
        if self.store_traj:
            self.x_traj = np.zeros((self.nx + self.ne + 2 * self.ni, max_iter + 1))
            self.x_traj[:, 0] = np.hstack((self.x, self.l, self.mu, self.s))
            
        self.p = None
        
        
    def get_residual(self, tau):
        r_L = self.Q @ self.x + self.c + self.A.T @ self.l + self.C.T @ self.u
        r_e = self.A @ self.x - self.b
        r_i = self.C @ self.x - self.d + self.s
        r_c = self.u * self.s - tau
        
        r = np.hstack([r_L, r_e, r_i, r_c])
        return r
        
    
    def get_x_sol_cvxpy(self):
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
    
    
    def compute_step(self, solver: LinSysSolver, tau: float):
        self.p = solver.solve(self.Q, self.A, self.C, self.s, self.mu, self.get_residual(tau))
        return self.p
    
    
    def execute_step(self, alpha):
        if self.p is None:
            raise Exception("No step size has been computed for the current iterate.")
        self.x += alpha * self.p[:self.idx_l]
        self.l += alpha * self.p[self.idx_l:self.idx_mu]
        self.mu += alpha * self.p[self.idx_mu:self.idx_su]
        self.s += alpha * self.p[self.idx_s:]
        
        self.p = None
        return
    
    
    def get_step_mu_s(self):
        if self.p is None:
            raise Exception("No step size has been computed for the current iterate.")
        return np.split(self.p[self.idx_mu:], [self.ni])
    
    def get_equality_constraint(self):
        # A = np.ndarray((0, nx)) if (A is None) else A
        # C = np.ndarray((0, nx)) if (C is None) else C
        
        # b = np.ndarray((0,)) if (b is None) else b
        # d = np.ndarray((0,)) if (d is None) else d
        pass
    
    def get_x_init(self):
        #TODO: move into solver class and use heuristic from OOQP
        return np.zeros(self.nx)
    
    def get_l_init(self):
        #TODO: move into solver class and use heuristic from OOQP
        return 10 * np.ones(self.ne)
    
    def get_mu_init(self):
        #TODO: move into solver class and use heuristic from OOQP
        return 10 * np.ones(self.ni)
    
    def get_s_init(self):
        #TODO: move into solver class and use heuristic from OOQP
        return 10 * np.ones(self.ni)
