import numpy as np
from osqp_benchmarks.random_qp import RandomQPExample
from osqp_benchmarks.control import ControlExample
from scipy.sparse.linalg import ArpackNoConvergence
from cvxpy import SolverError
from linsys_solver import LinSysSolver


class ConvexQP():
    def __init__(self, n: int, seed=1) -> None:
        """
        Class interfacing the OSQP benchmark suite. Loads a problem from the OSQP benchmark suite and holds the matrices
        and vectors of the QP in the format as specified for this solver.
        n: dimension of decision variable
        seed: for random problem generation
        """
        self.Q, self.c = self.get_cost()
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
        
        # current step direction for the QP
        self.p = None
        
    def get_residual(self, tau):
        """
        Compute the residuals of the KKT system for the current iterate in the primal and dual variables and the given
        smoothening parameter tau.
        """
        r_L = self.Q @ self.x + self.c + self.A.T @ self.l + self.C.T @ self.mu
        r_e = self.A @ self.x - self.b
        r_i = self.C @ self.x - self.d + self.s
        r_c = self.mu * self.s - tau
        
        r = np.hstack([r_L, r_e, r_i, r_c])
        return r
    
    def get_x_sol_cvxpy(self):
        """
        Get the solution to the QP as obtained through CVXPY (as kind of ground truth reference)
        """
        try:
            self.osqp.cvxpy_problem.solve()
        except (ArpackNoConvergence, SolverError):
            return False, None
        return True, self.osqp.revert_cvxpy_solution()[0]
    
    def get_equality_constraint(self):
        """
        Returns: Matrix A and vector b of equality constraint from the OSQP benchmark problem instance.
        """
        raise NotImplementedError("Must be implemented by derived class of ConvexQP.")
    
    def get_cost(self):
        """
        Returns: Matrix Q and vector c defining the QP cost term from the OSQP benchmark problem instance.
        """
        raise NotImplementedError("Must be implemented by derived class of ConvexQP.")
    
    def get_inequality_constraint(self):
        """
        Returns: Matrix C and vector d defining the inequality constraint from the OSQP benchmark problem instance.
        """
        raise NotImplementedError("Must be implemented by derived class of ConvexQP.")
    
    def compute_step(self, solver: LinSysSolver, tau: float):
        """
        Compute step direction for the current primal and dual iterates and current smoothening parameter tau.
        """
        self.p = solver.solve(self.Q, self.A, self.C, self.s, self.mu, self.get_residual(tau))
        return self.p
    
    def execute_step(self, alpha):
        """
        Follow the current step direction for the fraction alpha.
        """
        if self.p is None:
            raise Exception("No step size has been computed for the current iterate.")
        self.x += alpha * self.p[:self.idx_l]
        self.l += alpha * self.p[self.idx_l:self.idx_mu]
        self.mu += alpha * self.p[self.idx_mu:self.idx_s]
        self.s += alpha * self.p[self.idx_s:]
        
        self.p = None
        return
    
    def get_step_mu_s(self):
        """
        Returns: step directions for variables (mu, s)
        """
        if self.p is None:
            raise Exception("No step size has been computed for the current iterate.")
        return np.split(self.p[self.idx_mu:], [self.ni])
    
    def get_x_init(self):
        """
        Initial value for iterates of primal variable x.
        """
        return np.zeros(self.nx)
    
    def get_l_init(self):
        """
        Initial value for iterates of dual variable lambda.
        """
        return 10 * np.ones(self.ne)
    
    def get_mu_init(self):
        """
        Initial value for iterates of dual variable mu.
        """
        return 10 * np.ones(self.ni)
    
    def get_s_init(self):
        """
        Initial value for iterates of auxiliary variable s.
        """
        return 10 * np.ones(self.ni)
    
    def get_true_sparsity(self):
        """
        retrospectively not very useful for analysis. Functions giving information on sparsitiy of KKT
        system matrices are more informative.
        """
        non_zeros = np.count_nonzero(self.Q) + np.count_nonzero(self.A) + np.count_nonzero(self.C)
        true_sparsity = non_zeros / (self.nx * (self.nx + self.ne + self.ni))
        return true_sparsity
    
    def get_M_sparsity(self):
        """
        Compute sparsity of full KKT system of QP (without elimination of any variables)
        """
        non_zeros = np.count_nonzero(self.Q) + 2 * (np.count_nonzero(self.A) + np.count_nonzero(self.C))\
            + 3 * self.ni
        sparsity = non_zeros / (self.nx + self.ne + 2 * self.ni)**2
        return sparsity
    
    def get_M_sym_sparsity(self):
        """
        Compute sparsity of condensed KKT system of QP (after eliminating mu and s)
        """
        non_zeros = np.count_nonzero(self.Q + self.C.T @ self.C) + 2 * np.count_nonzero(self.A)
        sparsity = non_zeros / (self.nx + self.ne)**2
        return sparsity
    

class RandomQP(ConvexQP):
    def __init__(self, n: int, seed=1, sparsity=0.15) -> None:
        """
        Interfaces to the adapted version of the RandomQP class from the OSQP benchmark suite.
        Parameters:
            n: dimensionality decision variable x.
            seed: seeding random problem generation.
            sparsity: defines sparsity of equality and inequality constraint matrices A and C and influences
            sparsity of cost matrix Q.
        """
        self.sparsity = sparsity
        self.osqp = RandomQPExample(n, seed, sparsity)
        
        super().__init__(n, seed)
    
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
    def __init__(self, n: int, seed=1) -> None:
        """
        Interfaces to the Control Example class from the OSQP benchmark suite.
        Parameters:
            n: dimensionality decision variable x.
            seed: seeding random problem generation.
        """
        self.osqp = ControlExample(n)
        
        super().__init__(n, seed)
        
        
    def get_cost(self):
        Q = self.osqp.qp_problem['P'].toarray()
        c = self.osqp.qp_problem['q']
        return Q, c
    
    def get_equality_constraint(self):
        m = self.osqp.qp_problem['m']   # no of constraints
        n = self.osqp.qp_problem['n']   # no of variables (x & u over [0, T])
        
        m_e = m - n     # upper and lower bounds per variables, remainder are system dynamics
        
        A = self.osqp.qp_problem['A'].toarray()[:m_e]
        b = self.osqp.qp_problem['l'][:m_e]
        
        return A, b
    
    def get_inequality_constraint(self):
        m = self.osqp.qp_problem['m']   # no of constraints
        n = self.osqp.qp_problem['n']   # no of variables (x & u over [0, T])
        
        m_e = m - n     # upper and lower bounds per variables, remainder are system dynamics
        
        C = self.osqp.qp_problem['A'].toarray()[m_e:]
        d_l = self.osqp.qp_problem['l'][m_e:]
        d_u = self.osqp.qp_problem['u'][m_e:]
        
        C = np.vstack((C, -C))
        d = np.hstack((d_u, -d_l))
        
        return C, d
