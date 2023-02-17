import numpy as np
from matplotlib import pyplot as plt
from timeit import default_timer as timer
from plot_2d import plot_2d
from ldlt_solver import LDLTSolverEigen, LDLTSolverOwn, NumpySolver


EMPTY = np.ndarray((0,))


class IPSolver():
    def __init__(self, name) -> None:
        self.linear_system = None
        self.name = name
    
    def solve_QP(self, Q: np.ndarray, c: np.ndarray, x_init: np.ndarray, tol_r: float=1e-8, A: np.ndarray=None,
                 b: np.ndarray=EMPTY, C: np.ndarray=None, d: np.ndarray=EMPTY, l_init: np.ndarray=EMPTY,
                 u_init: np.ndarray=EMPTY, s_init: np.ndarray=EMPTY, t_init: float=0.75, red_t: float=0.3,
                 tol_t: float=1e-8, red_alpha: float=0.9, max_iter: int=100, min_alpha: float=1e-8):
        x_k = x_init
        if l_init is not None:
            l_k = l_init
        if u_init is not None:
            u_k = u_init
        if s_init is not None:
            s_k = s_init
        tau = t_init
        
        nx = len(x_init)
        ne = len(l_init) if l_init is not None else 0
        ni = len(u_init) if u_init is not None else 0
        
        solver = LDLTSolverEigen(nx, ne, ni)
        
        A = np.ndarray((0, nx)) if (A is None) else A
        C = np.ndarray((0, nx)) if (C is None) else C
        
        b = np.ndarray((0,)) if (b is None) else b
        d = np.ndarray((0,)) if (d is None) else d
            
        x_traj = np.zeros((nx + ne + 2 * ni, max_iter + 1))
        x_traj[:, 0] = np.hstack((x_init, l_init, u_init, s_init))
        iter_sol = 0
        for i in range(max_iter):
            # compute the residuals
            r_L = Q @ x_k + c + A.T @ l_k + C.T @ u_k
            r_e = A @ x_k - b
            r_i = C @ x_k - d + s_k
            r_c = u_k * s_k - tau
            
            r = np.hstack([r_L, r_e, r_i, r_c])

            iter_sol = i
            if np.max(r) <= tol_r:
                # only if there are inequalities we need to reduce tau
                if len(u_k) > 0:
                    if tau <= tol_t:
                        break
                    else:
                        # print('reducing tau')
                        tau *= red_t
                else:
                    # print(f"Solution found after {i} iterations.")
                    break

            dz = solver.solve(Q, A, C, s_k, u_k, r)
                        
            dx, dl, du, ds = np.split(dz, [nx, nx + ne, nx + ne + ni])
            
            alpha = 1
            # only if there are inequality constraints:
            if len(u_k > 0):
                while alpha >= min_alpha:
                    u_t = u_k + alpha * du
                    s_t = s_k + alpha * ds
                    if np.all(u_t >= 0.05 * np.min(u_k)) and np.all(s_t >= 0.05 * np.min(s_k)):
                        break
                    alpha *= red_alpha
                if alpha < min_alpha:
                    raise('No valid step length found.')
                
            x_k += alpha * dx
            l_k += alpha * dl
            u_k += alpha * du
            s_k += alpha * ds
            
            z_k = np.hstack((x_k, l_k, u_k, s_k))
            
            x_traj[:, i+1] = z_k
            
        print(f"Solution found in {iter_sol} iterations")
        
        return x_traj[:, :iter_sol+1]

# Q = np.array([[4, 0],
#               [0, 1]])
# c = np.array([-2, -2])
Q = np.zeros((2, 2))
c = np.array([1, 1])
# c = np.array([0, 0])


A = np.array([[1, -1]], np.float64)
b = np.array([0], np.float64)

C = -1 * np.array([[-1, 3],
                   [4, 1]])
d = np.array([-3, -4])
u_init = np.array([10, 10], np.float64)
s_init = np.array([10, 10], np.float64)
# C = -1 * np.array([[1, 1]])
# d = np.array([-6])
# u_init = np.array([10], np.float64)
# s_init = np.array([10], np.float64)

x_init = np.array([-15, 15], np.float64)
l_init = np.array([10], np.float64)

t_init = 0.25
tol_r = 1e-8
tol_t = 1e-8
red_alpha = 0.9
red_t = 1 / 3

# # solver = IPSolver("test").solve_QP(Q, c, A, b, C, d, x_init, l_init, u_init, s_init, t_init, red_alpha, red_t, tol_r, tol_t, 500, 1e-8)
# solver = IPSolver("test")
# start = timer()
# res = solver.solve_QP(Q, c, x_init, tol_r, C=C, d=d, u_init=u_init, s_init=s_init, t_init=t_init, max_iter=100)
# end = timer()
# print(end - start)
# # res = solver.solve_QP(Q, c, x_init, tol_r, A=A, b=b, l_init=l_init, t_init=t_init, max_iter=100)
# # print(res)
# plot_2d(Q, c, res[:2], -10, 5, -10, 5, C=C, d=d)
# # plot_2d(Q, c, res[:2], -10, 5, -10, 5, A=A, b=b)
