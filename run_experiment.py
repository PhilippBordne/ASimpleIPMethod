from qp_loader import RandomQP, ControlQP
from ip_solver import IPSolver
from ldlt_solver import LDLTSolverEigen, LUSolverNumpy
from plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from scipy.sparse.linalg import ArpackNoConvergence

n = 100
sparse_M_sym_08 = np.zeros(10)
sparse_M_08 = np.zeros(10)
sparse_M_sym_15 = np.zeros(10)
sparse_M_15 = np.zeros(10)
sparse_M_sym_30 = np.zeros(10)
sparse_M_30 = np.zeros(10)
for i in range(10):
    qp = RandomQP(n, i, sparsity=0.08)
    sparse_M_08[i] = qp.get_M_sparsity()
    sparse_M_sym_08[i] = qp.get_M_sym_sparsity()
    qp = RandomQP(n, i, sparsity=0.15)
    sparse_M_15[i] = qp.get_M_sparsity()
    sparse_M_sym_15[i] = qp.get_M_sym_sparsity()
    qp = RandomQP(n, i, sparsity=0.3)
    sparse_M_30[i] = qp.get_M_sparsity()
    sparse_M_sym_30[i] = qp.get_M_sym_sparsity()

print(f"rho=0.08  M: mean {np.mean(sparse_M_08)} std {np.std(sparse_M_08)}")
print(f"rho=0.08  M_sym: mean {np.mean(sparse_M_sym_08)} std {np.std(sparse_M_sym_08)}")
print(f"rho=0.15  M: mean {np.mean(sparse_M_15)} std {np.std(sparse_M_15)}")
print(f"rho=0.15  M_sym: mean {np.mean(sparse_M_sym_15)} std {np.std(sparse_M_sym_15)}")
print(f"rho=0.30  M: mean {np.mean(sparse_M_30)} std {np.std(sparse_M_30)}")
print(f"rho=0.30  M_sym: mean {np.mean(sparse_M_sym_30)} std {np.std(sparse_M_sym_30)}")

# # qp = RandomQP(40, seed=1)
# qp = RandomQP(30, seed=6, sparsity=0.09)
# print(qp.get_x_sol_cvxpy())
# # solver = LUSolverNumpy(qp.nx, qp.ne, qp.ni)
# # ip_method = IPSolver(qp, solver)

# plt.spy(qp.Q)
# plt.show()

# # plt.spy(qp.A)
# # plt.show()

# while not ip_method.verify_convergence() and not ip_method.reached_iteration_limit():
#     ip_method.solver_step()
    
# print(f"Problem converged: {ip_method.verify_convergence()}")
# print(f"In {ip_method.iter} iterations.")

# # print('Solution CVXPY')
# # print(qp.get_x_sol_cvxpy())

# print('Solution IPMethod')
# print(qp.x)

# # print("Difference")
# # print(qp.x - qp.get_x_sol_cvxpy())
