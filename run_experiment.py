from qp_loader import RandomQP, ControlQP
from ip_solver import IPSolver
from ldlt_solver import LDLTSolverEigen
from plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt


# qp = RandomQP(40, seed=1)
qp = ControlQP(4, seed=1)
solver = LDLTSolverEigen(qp.nx, qp.ne, qp.ni)
ip_method = IPSolver(qp, solver)

plt.spy(qp.Q[10 * 4:10 * 4+8,10 * 4:10 * 4+8])
plt.show()

plt.spy(qp.A)
plt.show()

while not ip_method.verify_convergence() and not ip_method.reached_iteration_limit():
    ip_method.solver_step()
    
print(f"Problem converged: {ip_method.verify_convergence()}")
print(f"In {ip_method.iter} iterations.")

print('Solution CVXPY')
print(qp.get_x_sol_cvxpy())

print('Solution IPMethod')
print(qp.x)

print("Difference")
print(qp.x - qp.get_x_sol_cvxpy())
