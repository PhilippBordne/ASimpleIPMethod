from qp_loader import ConvexQP
from ip_solver import IPSolver
from plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer


qp = ConvexQP(100, seed=1)

Q = qp.Q
c = qp.c
C = qp.C
d = qp.d

ni = len(C)
nx = len(Q)

solver = IPSolver('some')
print(ni)
start = timer()
# res = solver.solve_QP(Q=Q, c=c, C=C, d=d, x_init=np.zeros(nx), u_init=np.ones(ni), s_init=np.ones(ni), max_iter=400, tol_r=1e-6, tol_t=1e-6)
# res = solver.solve_QP(Q=Q, c=c, C=C, d=d, x_init=np.zeros(nx, dtype=np.float128), u_init=np.ones(ni, dtype=np.float128), s_init=np.ones(ni, dtype=np.float128), max_iter=400)
res = solver.solve_QP(Q=Q, c=c, C=C, d=d, x_init=np.zeros(nx), u_init=np.ones(ni), s_init=np.ones(ni), max_iter=400)
end = timer()
print(end - start)

print('Solution CVXPY')
print(qp.get_x_sol_cvxpy())

print('Solution IPMethod')
print(res[:nx, -1])

# plot_2d(Q=Q, c=c, C=c, d=d, x_traj=res[:2], x_min=-10, x_max=10, y_min=-10, y_max=10)
