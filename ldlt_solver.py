import numpy as np
from scipy.linalg import ldl 

class LDLT_solver():
    def __init__(self, nx: int, ne: int, ni: int) -> None:
        self.nx = nx
        self.ne = ne
        self.ni = ni
        self.idx_x = 0
        self.idx_e = nx
        self.idx_i = nx + ne
        self.idx_c = nx + ne + ni
    
    def symmetrize(self, Q, A, C, s, mu, r):
        S_inv = np.diag(1 / s)
        U = np.diag(mu)
        
        # if self.ne > 0:
        #     M = np.vstack((np.hstack((Q + C.T @ S_inv @ U @ C, A.T)),
        #                   np.hstack((A, np.zeros((self.ne, self.ne))))))
        # else:
        #     M = Q + C.T @ S_inv @ U @ C
        M = np.vstack((np.hstack((Q + C.T @ S_inv @ U @ C, A.T)),
                        np.hstack((A, np.zeros((self.ne, self.ne))))))
        
        r = np.concatenate((r[:self.nx] + C.T @ S_inv @ (U @ r[self.idx_i:self.idx_c] - r[self.idx_c:]),
                            r[self.idx_e:self.idx_i]))
               
        return M, r
        
    def recover_step(self, Q, A, C, s, mu, r, delta_x, delta_l):
        S = np.diag(s)
        S_inv = np.diag(1 / s)
        U = np.diag(mu)
        U_inv = np.diag(1 / mu)
        
        delta_mu = S_inv @ (U @ (r[self.idx_i:self.idx_c] + C @ delta_x) - r[self.idx_c:])
        delta_s = - U_inv @ (r[self.idx_c:] + S @ delta_mu)
        
        return np.concatenate((delta_x, delta_l, delta_mu, delta_s))
    
    def solve_ls(self, M, r):
        # get L, D matrix and row permutation
        L, D, perm = ldl(M)
        n = self.nx + self.ne
        # build permutation matrix from perm
        # perm_matrix = np.zeros((n, n))
        # perm_matrix[np.arange(n), perm] = 1
        
        # convert residual to RHS and match the row permutation of M
        # perform all computations in-place
        z = - r[perm]
        
        # backsolve for Lz = r, z prior is r, z posterior is z
        for i in range(n):
            z[i] = z[i] - np.sum(L[i, :i] * z[:i])
        
        # solve for y from Dy = z (z prior is z, posterior is y)
        z = 1 / np.diag(D) * z
        
        # get actual step p for L^T @ p = y (z prior is y, z posterior is p)
        for i in reversed(range(n)):
            z[i] = z[i] - np.sum(L[i+1:, i] * z[i+1:])
        
        return z
    
    def solve_via_LDLT(self, Q, A, C, s, mu, r):
        M_sym,  r_sym = self.symmetrize(Q, A, C, s, mu, r)
        
        delta_x, delta_l = np.split(self.solve_ls(M_sym, r_sym), [self.nx])
        step = self.recover_step(Q, A, C, s, mu, r, delta_x, delta_l)
        
        return step
    