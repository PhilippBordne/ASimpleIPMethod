"""
A simple example on how to use a IP method together with the chosen solver on a QP problem instance.
"""

from ip_method import IPMethod
from linsys_solver import LDLTSolverOwn
from qps import RandomQP

# create a QP to solve and a solver to solve its KKT systems.
qp = RandomQP(60, seed=2)
solver = LDLTSolverOwn(qp.nx, qp.ne, qp.ni)

# create an IP method instance acting on the QP and using the defined linear system solvers
ip_solver = IPMethod(qp, solver)

# perform steps of the ip method until residuals and tau are within tolerance or the iteration limit
# was reached.
while not ip_solver.verify_convergence() and not ip_solver.reached_iteration_limit():
    ip_solver.step()

print("###FINAL###")
# get number of iterations for solving the problem
print(ip_solver.iter)
# get optimal primal variable
print(qp.x)
# get reference solution as obtained from CVXPY
print(qp.get_x_sol_cvxpy())
