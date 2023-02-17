import numpy as np
from scipy.linalg import ldl
from matplotlib import pyplot as plt
from eigenpy import LDLT

#TODO: Consider making LinSysSolver independent from problem dimensions
class LinSysSolver():
    def __init__(self, nx: int, ne: int, ni: int) -> None:
        self.nx = nx
        self.ne = ne
        self.ni = ni
        self.idx_x = 0
        self.idx_e = nx
        self.idx_i = nx + ne
        self.idx_c = nx + ne + ni
        
    def prepare_lin_sys(self, Q, A, C, s, mu, r):
        raise NotImplementedError(f"symmetrize method has to be implemented by a derived class.")
    
    def recover_step(self, Q, A, C, s, mu, r, delta_p):
        raise NotImplementedError(f"recover_step method has to be implemented by a derived class.")        
    
    def solve(self, Q, A, C, s, mu, r):
        M_sym,  r_sym = self.prepare_lin_sys(Q, A, C, s, mu, r)
        
        delta_p = self.solve_lin_sys(M_sym, r_sym)
        step = self.recover_step(Q, A, C, s, mu, r, delta_p)
        
        return step
       

class LDLTSolver(LinSysSolver):
    def prepare_lin_sys(self, Q, A, C, s, mu, r):
        S_inv = np.diag(1 / s)
        U = np.diag(mu)
        
        M = np.vstack((np.hstack((Q + C.T @ S_inv @ U @ C, A.T)),
                        np.hstack((A, np.zeros((self.ne, self.ne))))))
        
        r = np.concatenate((r[:self.nx] + C.T @ S_inv @ (U @ r[self.idx_i:self.idx_c] - r[self.idx_c:]),
                            r[self.idx_e:self.idx_i]))
               
        return M, r
    
    def recover_step(self, Q, A, C, s, mu, r, delta_p):
        S = np.diag(s)
        S_inv = np.diag(1 / s)
        U = np.diag(mu)
        U_inv = np.diag(1 / mu)
        
        delta_x, delta_l = np.split(delta_p, [self.nx])
        
        delta_mu = S_inv @ (U @ (r[self.idx_i:self.idx_c] + C @ delta_x) - r[self.idx_c:])
        delta_s = - U_inv @ (r[self.idx_c:] + S @ delta_mu)
        
        return np.concatenate((delta_x, delta_l, delta_mu, delta_s))
    

class LDLTSolverEigen(LDLTSolver):
    def solve_lin_sys(self, M, r):
        decomp = LDLT(M)
        return decomp.solve(-r)
    
    
class LDLTSolverOwn(LDLTSolver):
    def solve_lin_sys(self, M, r):
        L, D, perm = ldl(M)
        
        n = len(M)

        # convert residual to RHS and match the row permutation of M
        # perform all computations in-place
        # z = - np.array(r[perm], dtype=np.float128)
        z = - r[perm]
        
        # backsolve for Lz = r, z prior is r, posterior is z
        for i in range(n):
            z[i] = z[i] - np.sum(L[i, :i] * z[:i])
        
        # solve for y from Dy = z (z prior is z, posterior is y)
        z = 1 / np.diag(D) * z
        
        # get actual step p for L^T @ p = y (z prior is y, z posterior is p)
        for i in reversed(range(n)):
            z[i] = z[i] - np.sum(L[i+1:, i] * z[i+1:])
        
        return z
    

class NumpySolver(LinSysSolver):
    def prepare_lin_sys(self, Q, A, C, s, mu, r):
        # compute the linear system matrix
        J_L = np.hstack((Q, A.T, C.T, np.zeros((self.nx, self.ni))))
        J_e = np.hstack((A, np.zeros((self.ne, self.ne + 2 * self.ni))))
        J_i = np.hstack((C, np.zeros((self.ni, self.ne + self.ni)), np.eye(self.ni)))
        J_c = np.hstack((np.zeros((self.ni, self.nx + self.ne)), np.diag(s), np.diag(mu)))
        M = np.vstack((J_L, J_e, J_i, J_c))
        
        return M, r
    
    def solve_lin_sys(self, M, r):
        return np.linalg.solve(M, -r)
    
    def recover_step(self, Q, A, C, s, mu, r, delta_p):
        return delta_p
    

    
class LDLT_solver_multiply(LinSysSolver):
    def __init__(self, nx: int, ne: int, ni: int) -> None:
        super().__init__(nx, ne, ni)
    
    def prepare_lin_sys(self, Q, A, C, s, mu, r):
        # S_inv = np.diag(1 / s)
        S = np.diag(s)
        U = np.diag(mu)
        
        M = np.vstack((np.hstack((Q, A.T, C.T, np.zeros((self.nx, self.ni)))),
                       np.hstack((A, np.zeros((self.ne, self.ne + 2 * self.ni)))),
                       np.hstack((C, np.zeros((self.ni, self.ne + self.ni)), S)),
                       np.hstack((np.zeros((self.ni, self.nx + self.ne)), S, U @ S))
                       ))
        
        return M, r
        
    def recover_step(self, Q, A, C, s, mu, r, delta_p):
        delta_p[self.idx_c:] *= s
        
        return delta_p
    