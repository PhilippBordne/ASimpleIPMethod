import numpy as np
from scipy.linalg import ldl
from scipy import sparse
import eigenpy
from qps import ConvexQP
from matplotlib import pyplot as plt

class KKTSysSolver():
    def __init__(self) -> None:
        """
        Abstract class that defines the linear system solve for the KKT system of a Quadratic Program.
        """
        pass
    
    def solve(self, qp: ConvexQP, tau):
        """
        Execute the 3 steps that are involved when solving the KKT system of a QP:
        1. prepare the linear system (from residuals and system matrices), might eliminate some variables
        2. solve the linear system using the linsys solution method of the class (LU or LDLT)
        3. in case variables whrere eliminated from KKT system recover the full step.
        Set the computed step length in the qp
        """
        r = qp.get_residual(tau)
        M_sym,  r_sym = self.prepare_lin_sys(qp, r)
        delta_p = self.solve_lin_sys(M_sym, r_sym)
        step = self.recover_step(qp, delta_p, r)
        qp.set_p(step)
        
        return step
        
    def prepare_lin_sys(self, qp: ConvexQP, r):
        """
        Builds the KKT system from the QPs matrices and current iterate of primal and dual variables and residuals.
        """
        raise NotImplementedError(f"symmetrize method has to be implemented by a derived class.")
    
    def recover_step(self, qp: ConvexQP, delta_p, r):
        """
        Recovers step for QP in all primal and dual variables in case some of the where eliminated to obtain a
        condensed KKT system.
        """
        raise NotImplementedError(f"recover_step method has to be implemented by a derived class.")           

class LDLTSolver(KKTSysSolver):
    def prepare_lin_sys(self, qp: ConvexQP, r):
        """
        Eliminate mu and s and build a condensed KKT system
        """
        idx_e = qp.nx
        idx_i = idx_e + qp.ne
        idx_c = idx_i + qp.ni
        
        S_inv = np.diag(1 / qp.s)
        U = np.diag(qp.mu)
        
        M = np.vstack((np.hstack((qp.Q + qp.C.T @ S_inv @ U @ qp.C, qp.A.T)),
                        np.hstack((qp.A, np.zeros((qp.ne, qp.ne))))))
        
        r = np.concatenate((r[:idx_e] + qp.C.T @ S_inv @ (U @ r[idx_i:idx_c] - r[idx_c:]),
                            r[idx_e:idx_i]))
               
        return M, r
    
    def recover_step(self, qp: ConvexQP, delta_p, r):
        """
        Compute step for mu and s from the steps in x and lambda from the condensed KKT system and set the
        step stored with the QP.
        """
        idx_e = qp.nx
        idx_i = idx_e + qp.ne
        idx_c = idx_i + qp.ni
        
        S = np.diag(qp.s)
        S_inv = np.diag(1 / qp.s)
        U = np.diag(qp.mu)
        U_inv = np.diag(1 / qp.mu)
        
        delta_x, delta_l = np.split(delta_p, [qp.nx])
        
        delta_mu = S_inv @ (U @ (r[idx_i:idx_c] + qp.C @ delta_x) - r[idx_c:])
        delta_s = - U_inv @ (r[idx_c:] + S @ delta_mu)
        
        return np.concatenate((delta_x, delta_l, delta_mu, delta_s))
    

class LDLTSolverEigen(LDLTSolver):
    def solve_lin_sys(self, M, r):
        """
        Solve symmetric linear system using LDLT factorization through the implementation in the eigenpy / Eigen
        package.
        """
        decomp = eigenpy.LDLT(M)
        return decomp.solve(-r)
    
    
class LDLTSolverScipy(LDLTSolver):
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
        # unfortunately need to use inverse here because D might only be block diagonal of unknown block size
        # p = 1 / np.diag(D) * p
        p = np.linalg.inv(D) @ p
        
        # p_3: backsubstitute to obtain RHS for term in brackets
        # L_tilde.T @ (P @ p) = L_tilde.T @ P @ p
        for i in reversed(range(n)):
            p[i] = p[i] - np.sum((L_tilde.T)[i, i+1:] * p[i+1:])
        
        # p_4: undo permutation to obtain target value p
        p = P.T @ p
        return p
    
class LUSolver(KKTSysSolver):
    def prepare_lin_sys(self, qp: ConvexQP, r):
        """
        Construct the full KKT system without any elimination of variables.
        """
        # compute the linear system matrix
        J_L = np.hstack((qp.Q, qp.A.T, qp.C.T, np.zeros((qp.nx, qp.ni))))
        J_e = np.hstack((qp.A, np.zeros((qp.ne, qp.ne + 2 * qp.ni))))
        J_i = np.hstack((qp.C, np.zeros((qp.ni, qp.ne + qp.ni)), np.eye(qp.ni)))
        J_c = np.hstack((np.zeros((qp.ni, qp.nx + qp.ne)), np.diag(qp.s), np.diag(qp.mu)))
        M = np.vstack((J_L, J_e, J_i, J_c))
        
        return M, r
    
    def recover_step(self, qp: ConvexQP, delta_p, r):
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
     