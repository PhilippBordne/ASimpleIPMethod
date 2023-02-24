import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from plot_2d import plot_2d
from linsys_solver import LinSysSolver
from qps import ConvexQP


class IPMethod():
    def __init__(self, qp: ConvexQP, solver: LinSysSolver, tol_r: float=1e-8, t_init: float=0.75, red_t: float=0.3,
                 tol_t: float=1e-8, beta: float=0.9, min_alpha: float=1e-8, max_iter: int=100) -> None:
        """
        Interior Point method tho minimize a convex quadratic programm with the option to use different linear system
        solvers for the KKT system.
        Params:
            qp: Convex Quadratic Program to solve
            solver: linear system solver for the KKT system
            tol_r: tolerance for the maximum residual of KKT system to deem the QP to have converged to its optimum.
            t_init: initial value of tau to smoothen the complementarity condition
            red_t: factor to reduce tau after KKT system was solved for current tau
            tol_t: tolerance for final value of tau after which solved KKT system is deemed as final solution
            beta: reduction factor for line search on step length to comply to non-negativity condition for mu and s
            min_alpha: value of alpha that causes no valid step length found error.
            max_iter: maximum number of iterations taken by the IP method.
        """
        self.qp = qp
        self.solver = solver
        self.tol_r = tol_r
        
        self.tau = t_init
        self.red_t = red_t
        self.tol_t = tol_t
        
        self.red_alpha = beta
        self.min_alpha = min_alpha
        
        self.max_iter = max_iter
        self.iter = 0
    
    def step(self):
        """
        Compute step for root finding of KKT system, perform line search to fullfill non-negativity of mu and s
        and let qp execute step.
        """
        self.qp.compute_step(self.solver, self.tau)
        alpha = self.compute_step_length()
        self.qp.execute_step(alpha)
        self.iter += 1
    
    
    def verify_convergence(self):
        """
        Checks residual of KKT system, if residual within tolerance reduces tau if not below threshold.
        """
        r = self.qp.get_residual(self.tau)
        if np.max(r) <= self.tol_r:
            # only if there are inequalities we need to reduce tau
            if self.qp.ni > 0:
                if self.tau <= self.tol_t:
                    return True
                else:
                    self.tau *= self.red_t
            else:
                return True
        return False
    
    
    def compute_step_length(self):
        """
        Backtracking line search for current step of QP to fullfill non-negativity condition of s and mu.
        """
        alpha = 1
        dmu, ds = self.qp.get_step_mu_s() 
        # only if there are inequality constraints:
        if len(self.qp.mu) > 0:
            while alpha >= self.min_alpha:
                u_t = self.qp.mu + alpha * dmu
                s_t = self.qp.s + alpha * ds
                if np.all(u_t >= 0.05 * np.min(self.qp.mu)) and np.all(s_t >= 0.05 * np.min(self.qp.s)):
                    break
                alpha *= self.red_alpha
            if alpha < self.min_alpha:
                raise('No valid step length found.')
        return alpha
        
    
    def reached_iteration_limit(self):
        """
        Check if number of iterations has reached iteration limit.
        """
        return self.iter >= self.max_iter
