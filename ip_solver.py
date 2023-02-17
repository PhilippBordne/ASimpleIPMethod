import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from plot_2d import plot_2d
from ldlt_solver import LinSysSolver
from qp_loader import ConvexQP


class IPSolver():
    def __init__(self, qp: ConvexQP, solver: LinSysSolver, tol_r: float=1e-8, t_init: float=0.75, red_t: float=0.3,
                 tol_t: float=1e-8, red_alpha: float=0.9, max_iter: int=100, min_alpha: float=1e-8) -> None:
        self.qp = qp
        self.solver = solver
        self.tol_r = tol_r
        
        self.t = t_init
        self.red_t = red_t
        self.tol_t = tol_t
        
        self.red_alpha = red_alpha
        self.min_alpha = min_alpha
        
        self.max_iter = max_iter
        self.iter = 0
    
    def solver_step(self):
        self.qp.compute_step(self.solver, self.tau)
        alpha = self.compute_step_length()
        self.qp.execute_step(alpha)
    
    
    def verify_convergence(self):
        r = self.qp.get_residual(self.t)
        if np.max(r) <= tol_r:
            # only if there are inequalities we need to reduce tau
            if len(u_k) > 0:
                if tau <= tol_t:
                    return True
                else:
                    tau *= red_t
            else:
                return True
        return False
    
    
    def compute_step_length(self):
        alpha = 1
        dmu, ds = self.qp.get_step_mu_s() 
        # only if there are inequality constraints:
        if len(self.qp.mu) > 0:
            while alpha >= min_alpha:
                u_t = self.qp.mu + alpha * dmu
                s_t = self.qp.s + alpha * ds
                if np.all(u_t >= 0.05 * np.min(self.qp.mu)) and np.all(s_t >= 0.05 * np.min(self.qp.s)):
                    break
                alpha *= self.red_alpha
            if alpha < min_alpha:
                raise('No valid step length found.')
        return alpha
        
    
    def reached_iteration_limit():
        return self.iter >= self.max_iter
