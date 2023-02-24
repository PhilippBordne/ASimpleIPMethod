import numpy as np
from osqp_benchmarks.random_qp import RandomQPExample
from osqp_benchmarks.control import ControlExample
from scipy.sparse.linalg import ArpackNoConvergence
from cvxpy import SolverError
from ldlt_solver import LinSysSolver


class ConvexQP():
    def __init__(self, n: int, seed=1, sparsity=0.15, store_traj=False) -> None:
        """
        Class interfacing the OSQP benchmark suite. Loads a problem from the OSQP benchmark suit and holds the matrices
        and vectors of the QP in the format as specified for this solver.
        n: dimension of decision variable
        """
        
        self.Q, self.c = self.get_cost()
        # self.Q = self.osqp.P.toarray()
        # self.c = self.osqp.q
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
        r_L = self.Q @ self.x + self.c + self.A.T @ self.l + self.C.T @ self.mu
        r_e = self.A @ self.x - self.b
        r_i = self.C @ self.x - self.d + self.s
        r_c = self.mu * self.s - tau
        
        r = np.hstack([r_L, r_e, r_i, r_c])
        return r
    
    def get_x_sol_cvxpy(self):
        try:
            self.osqp.cvxpy_problem.solve()
        except (ArpackNoConvergence, SolverError):
            return False, None
        return True, self.osqp.revert_cvxpy_solution()[0]
    
    
    def get_equality_constraint(self):
        raise NotImplementedError("Must be implemented by derived class of ConvexQP.")
    
    def get_cost(self):
        raise NotImplementedError("Must be implemented by derived class of ConvexQP.")
    
    def get_inequality_constraint(self):
        raise NotImplementedError("Must be implemented by derived class of ConvexQP.")
    
    def compute_step(self, solver: LinSysSolver, tau: float):
        self.p = solver.solve(self.Q, self.A, self.C, self.s, self.mu, self.get_residual(tau))
        return self.p
    
    def execute_step(self, alpha):
        if self.p is None:
            raise Exception("No step size has been computed for the current iterate.")
        self.x += alpha * self.p[:self.idx_l]
        self.l += alpha * self.p[self.idx_l:self.idx_mu]
        self.mu += alpha * self.p[self.idx_mu:self.idx_s]
        self.s += alpha * self.p[self.idx_s:]
        
        self.p = None
        return
    
    def get_step_mu_s(self):
        if self.p is None:
            raise Exception("No step size has been computed for the current iterate.")
        return np.split(self.p[self.idx_mu:], [self.ni])
    
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
    
    def get_true_sparsity(self):
        # retrospectively not very useful for analysis. Functions giving information on sparsitiy of KKT
        # system matrices are more informative.
        non_zeros = np.count_nonzero(self.Q) + np.count_nonzero(self.A) + np.count_nonzero(self.C)
        true_sparsity = non_zeros / (self.nx * (self.nx + self.ne + self.ni))
        return true_sparsity
    
    def get_M_sparsity(self):
        non_zeros = np.count_nonzero(self.Q) + 2 * (np.count_nonzero(self.A) + np.count_nonzero(self.C))\
            + 3 * self.ni
        sparsity = non_zeros / (self.nx + self.ne + 2 * self.ni)**2
        return sparsity
    
    def get_M_sym_sparsity(self):
        non_zeros = np.count_nonzero(self.Q + self.C.T @ self.C) + 2 * np.count_nonzero(self.A)
        sparsity = non_zeros / (self.nx + self.ne)**2
        return sparsity
    

class RandomQP(ConvexQP):
    # self.__init__(self, n: int, seed=1, sparsity=0.15, store_traj=False) -> None:
    #     super().__init__()
    def __init__(self, n: int, seed=1, sparsity=0.15, store_traj=False) -> None:
        self.osqp = RandomQPExample(n, seed, sparsity)
        
        super().__init__(n, seed, sparsity, store_traj)
    
    def get_equality_constraint(self):
        A = self.osqp.A.toarray()[:self.osqp.ne]
        b = self.osqp.l[:self.osqp.ne]

        # get rid of all-zero rows
        non_zero_idx = []
        for i, row in enumerate(A):
            if not np.all(row==0):
                non_zero_idx += [i]
        return A[non_zero_idx], b[non_zero_idx]
    
    
    def get_inequality_constraint(self):
        C = self.osqp.A.toarray()[self.osqp.ne:]
        d = self.osqp.u[self.osqp.ne:]

        # get rid of all-zero rows
        non_zero_idx = []
        for i, row in enumerate(C):
            if not np.all(row==0):
                non_zero_idx += [i]
        return C[non_zero_idx], d[non_zero_idx]
    
    def get_cost(self):
        Q = self.osqp.P.toarray()
        c = self.osqp.q
        return Q, c
    

class ControlQP(ConvexQP):
    def __init__(self, n: int, seed=1, sparsity=0.15, store_traj=False) -> None:
        self.osqp = ControlExample(n)
        
        super().__init__(n, seed, sparsity, store_traj)
        
        
    def get_cost(self):
        Q = self.osqp.qp_problem['P'].toarray()
        c = self.osqp.qp_problem['q']
        return Q, c
    
    def get_equality_constraint(self):
        m = self.osqp.qp_problem['m']   # no of constraints
        n = self.osqp.qp_problem['n']   # no of variables (x & u)
        
        m_e = m - n     # upper and lower bounds per variables, remainder are system dynamics
        
        A = self.osqp.qp_problem['A'].toarray()[:m_e]
        b = self.osqp.qp_problem['l'][:m_e]
        
        return A, b
    
    def get_inequality_constraint(self):
        m = self.osqp.qp_problem['m']   # no of constraints
        n = self.osqp.qp_problem['n']   # no of variables (x & u)
        
        m_e = m - n     # upper and lower bounds per variables, remainder are system dynamics
        
        C = self.osqp.qp_problem['A'].toarray()[m_e:]
        d_l = self.osqp.qp_problem['l'][m_e:]
        d_u = self.osqp.qp_problem['u'][m_e:]
        
        C = np.vstack((C, -C))
        d = np.hstack((d_u, -d_l))
        
        return C, d
