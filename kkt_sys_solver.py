import numpy as np
from scipy.linalg import ldl
import eigenpy

#TODO: Consider making LinSysSolver independent from problem dimensions
class KKTSysSolver():
    def __init__(self, nx: int, ne: int, ni: int) -> None:
        """
        Abstract class that defines the linear system solve for the KKT system of a Quadratic Program.
        """
        self.nx = nx
        self.ne = ne
        self.ni = ni
        self.idx_x = 0
        self.idx_e = nx
        self.idx_i = nx + ne
        self.idx_c = nx + ne + ni
        
    def prepare_lin_sys(self, Q, A, C, s, mu, r):
        """
        Builds the KKT system from the QPs matrices and current iterate of primal and dual variables and residuals.
        """
        raise NotImplementedError(f"symmetrize method has to be implemented by a derived class.")
    
    def recover_step(self, Q, A, C, s, mu, r, delta_p):
        """
        Recovers step for QP in all primal and dual variables in case some of the where eliminated to obtain a
        condensed KKT system.
        """
        raise NotImplementedError(f"recover_step method has to be implemented by a derived class.")        
    
    def solve(self, Q, A, C, s, mu, r):
        """
        Execute the 3 steps that are involved when solving the KKT system of a QP:
        1. prepare the linear system (from residuals and system matrices), might eliminate some variables
        2. solve the linear system using the linsys solution method of the class (LU or LDLT)
        3. in case variables whrere eliminated from KKT system recover the full step.
        """
        M_sym,  r_sym = self.prepare_lin_sys(Q, A, C, s, mu, r)
        delta_p = self.solve_lin_sys(M_sym, r_sym)
        step = self.recover_step(Q, A, C, s, mu, r, delta_p)
        
        return step
       

class LDLTSolver(KKTSysSolver):
    def prepare_lin_sys(self, Q, A, C, s, mu, r):
        """
        Eliminate mu and s and build a condensed KKT system
        """
        S_inv = np.diag(1 / s)
        U = np.diag(mu)
        
        M = np.vstack((np.hstack((Q + C.T @ S_inv @ U @ C, A.T)),
                        np.hstack((A, np.zeros((self.ne, self.ne))))))
        
        r = np.concatenate((r[:self.nx] + C.T @ S_inv @ (U @ r[self.idx_i:self.idx_c] - r[self.idx_c:]),
                            r[self.idx_e:self.idx_i]))
               
        return M, r
    
    def recover_step(self, Q, A, C, s, mu, r, delta_p):
        """
        Compute step for mu and s from the steps in x and lambda from the condensed KKT system.
        """
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
        """
        Solve symmetric linear system using LDLT factorization through the implementation in the eigenpy / Eigen
        package.
        """
        decomp = eigenpy.LDLT(M)
        return decomp.solve(-r)
    
    
class LDLTSolverOwn(LDLTSolver):
    def solve_lin_sys(self, M, r):
        """
        Solve symmetric linear system by obtaining LDLT factorization from scipy and a backsubstitution procedure on the
        factorized linear system.
        """
        n = len(M)
        L, D, perm = ldl(M)
        
        # scipy.linalg.ldl returns L such that permutation needs to be applied first to make it triangular
        # create permutation matrix first
        P = np.zeros_like(M)
        for i, j in enumerate(perm):
            P[i, j] = 1

        # create references lower triangular matrix:
        L_tilde = P @ L
        
        # solve linear system for p via backsubstitution from:
        # L_tilde @ D @ L_tilde.T @ P @ p = - P @ r

        # convert residual to RHS and match the row permutation of M
        # perform all computations in-place
        p = - P @ r
        
        # p_1: backsubstitute to obtain RHS for term in brackets
        # L_tilde @ (D @ L_tilde.T @ P @ p) = - P @ r
        for i in range(n):
            p[i] = p[i] - np.sum(L_tilde[i, :i] * p[:i])
            
        # p_2: scale to obtain RHS for term in brackets
        # D @ (L_tilde.T @ P @ p) = D @ L_tilde.T @ P @ p
        p = 1 / np.diag(D) * p
        
        # p_3: backsubstitute to obtain RHS for term in brackets
        # L_tilde.T @ (P @ p) = L_tilde.T @ P @ p
        for i in reversed(range(n)):
            p[i] = p[i] - np.sum(L_tilde.T[i, i+1:] * p[i+1:])
        
        # p_4: undo permutation to obtain target value p
        p = P.T @ p
        return p
    
class LUSolver(KKTSysSolver):
    def prepare_lin_sys(self, Q, A, C, s, mu, r):
        """
        Construct the full KKT system without any elimination of variables.
        """
        # compute the linear system matrix
        J_L = np.hstack((Q, A.T, C.T, np.zeros((self.nx, self.ni))))
        J_e = np.hstack((A, np.zeros((self.ne, self.ne + 2 * self.ni))))
        J_i = np.hstack((C, np.zeros((self.ni, self.ne + self.ni)), np.eye(self.ni)))
        J_c = np.hstack((np.zeros((self.ni, self.nx + self.ne)), np.diag(s), np.diag(mu)))
        M = np.vstack((J_L, J_e, J_i, J_c))
        
        return M, r
    
    def recover_step(self, Q, A, C, s, mu, r, delta_p):
        """
        Nothing to do in LU case as we already computed the full step.
        """
        return delta_p

class LUSolverNumpy(LUSolver):
    def solve_lin_sys(self, M, r):
        """
        linear system solve via LU factorization as implemented in numpy (using LAPACK).
        """
        return np.linalg.solve(M, -r)
     