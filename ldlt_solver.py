import numpy as np
from scipy.linalg import ldl
from matplotlib import pyplot as plt
import eigenpy

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
        decomp = eigenpy.LDLT(M)
        return decomp.solve(-r)
    
    
class LDLTSolverOwn(LDLTSolver):
    def solve_lin_sys(self, M, r):
        n = len(M)
        L, D, perm = ldl(M)
        
        # scipy.linalg.ldl returns L such that permutation needs to be applied first to make it triangular
        # create permutation matrix first
        P = np.zeros_like(M)
        for i, j in enumerate(perm):
            P[i, j] = 1

        # if not np.all(perm == np.arange(len(M))):
        #     plt.spy(P@L)
        #     plt.show()
            
        # create references for upper and lower triangular matrices:
        L = P @ L
        U = L.T

        # convert residual to RHS and match the row permutation of M
        # perform all computations in-place
        z = - P @ r
        
        # z_1
        # backsolve for Lz = r, z prior is r, posterior is z
        # z_1 = np.zeros(n)
        for i in range(n):
            z[i] = z[i] - np.sum(L[i, :i] * z[:i])
            
        # z_2
        # solve for y from Dy = z (z prior is z, posterior is y)
        z = 1 / np.diag(D) * z
        
        # z_3
        # get actual step p for L^T @ p = y (z prior is y, z posterior is p)
        for i in reversed(range(n)):
            z[i] = z[i] - np.sum(U[i, i+1:] * z[i+1:])
        
        # delta p
        z = P.T @ z
        return z
    
class LUSolver(LinSysSolver):
    def prepare_lin_sys(self, Q, A, C, s, mu, r):
        # compute the linear system matrix
        J_L = np.hstack((Q, A.T, C.T, np.zeros((self.nx, self.ni))))
        J_e = np.hstack((A, np.zeros((self.ne, self.ne + 2 * self.ni))))
        J_i = np.hstack((C, np.zeros((self.ni, self.ne + self.ni)), np.eye(self.ni)))
        J_c = np.hstack((np.zeros((self.ni, self.nx + self.ne)), np.diag(s), np.diag(mu)))
        M = np.vstack((J_L, J_e, J_i, J_c))
        
        return M, r
    
    def recover_step(self, Q, A, C, s, mu, r, delta_p):
        return delta_p

class LUSolverNumpy(LUSolver):
    def solve_lin_sys(self, M, r):
        return np.linalg.solve(M, -r)
    
class LUSolverEigen(LUSolver):
    
    def solve_lin_sys(self, M, r):
        raise NotImplementedError("Appears there is no binding for LU factorization in EigenPy")
        decomp = eigenpy.LU(M)
        return decomp.solve(-r)
    