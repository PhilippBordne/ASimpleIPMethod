import numpy as np
from ip_solver import IPSolver
from ldlt_solver import LDLTSolverOwn
from qp_loader import RandomQP
from pathlib import Path

# results = np.array([[1, 3, np.inf]])

# np.savetxt(f'{Path(__file__).parent}/data/test.csv', results, delimiter=';')

qp = RandomQP(60, seed=2)
solver = LDLTSolverOwn(qp.nx, qp.ne, qp.ni)

ip_solver = IPSolver(qp, solver)

while not ip_solver.verify_convergence() and not ip_solver.reached_iteration_limit():
    ip_solver.solver_step()

print("###FINAL###")
print(ip_solver.iter)
print(qp.x)
print(qp.get_x_sol_cvxpy())
