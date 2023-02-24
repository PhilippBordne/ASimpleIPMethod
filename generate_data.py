"""
Script to benchmark performance of IP method employing different linear system solvers on instances
from the RandomQP and Control classes of the OSQP benchmark suite that differ in dimensionality and sparsity.
"""

import numpy as np
from pathlib import Path
import os
import time
from kkt_sys_solver import LDLTSolverEigen, LUSolverNumpy, LDLTSolverScipy
from qps import RandomQP, ControlQP
from ip_method import IPMethod

if not os.path.exists(Path(__file__).parent / 'data'):
    print('creatin directory')
    os.mkdir(Path(__file__).parent / 'data')
else:
    print("directory exits")

# classes to take the benchmark problems from
# classes = ["random_qp", "control"]
# classes = ["control"]
classes = ["random_qp"]

# linear system solvers to be evaluated together with the IP method
# solvers = ["LU", "LDLT_eigen", "LDLT_scipy"]
solvers = ["LDLT_scipy"]
# solvers = ["LDLT_eigen"]
# solvers = ["LU"]

# sparsity parameters for creation of the RandomQP instances
sparse = [0.08, 0.09, 0.1, 0.15, 0.3]

# dimensions for the random QP classes
dimensions_random = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# dimensions for the control QP classes
dimensions_control = [2, 4, 6, 8, 10]

# seeds to be used for random problem generation
seeds = np.arange(10)


for class_name in classes:
    if class_name == "random_qp":
        dims = dimensions_random
        sparsities = sparse
    elif class_name == "control":
        dims = dimensions_control
        sparsities = [1]  # no sparsity to set for control examples

        
    for solver_name in solvers:
        results = np.ndarray((0, 14))
        for dim in dims:
            for sparsity in sparsities:
                for seed in seeds:
                    if class_name == "random_qp":
                        qp = RandomQP(dim, seed, sparsity)
                    elif class_name == "control":
                        qp = ControlQP(dim, seed)
                    
                    # choose solver
                    if solver_name == "LU":
                        solver = LUSolverNumpy()
                    elif solver_name == "LDLT_eigen":
                        solver = LDLTSolverEigen()
                    elif solver_name == "LDLT_scipy":
                        solver = LDLTSolverScipy()
                    
                    # initialize IP method with solver and qp
                    ip_solver = IPMethod(qp, solver)
                    
                    print(f"on {class_name} using {solver_name}: dim {dim} | sparsity {sparsity} | seed {seed}")

                    start = time.process_time()  # measure CPU time in seconds
                    while not ip_solver.verify_convergence() and not ip_solver.reached_iteration_limit():
                        ip_solver.step()
                    end = time.process_time()
                    
                    # get data to be stored in the results for current problem instance
                    nx = qp.nx
                    ne = qp.ne
                    ni = qp.ni
                    
                    true_sparsity = qp.get_true_sparsity()
                    duration = end - start
                    iters = ip_solver.iter
                    
                    start = time.process_time()
                    cvxpy_converged, sol_x_cvxpy = qp.get_x_sol_cvxpy()
                    end = time.process_time()
                    duration_cvxpy = end - start
                    
                    if cvxpy_converged:
                        absdist_cvxpy_ip_sol = np.abs(sol_x_cvxpy - qp.x)
                        max_diff_cvxpy = np.max(absdist_cvxpy_ip_sol)
                        mean_diff_cvxpy = np.mean(absdist_cvxpy_ip_sol)
                    else:
                        max_diff_cvxpy = 0
                        mean_diff_cvxpy = 0
                        duration_cvxpy = np.inf
                    
                    ip_converged = ip_solver.verify_convergence()
                    
                    # store all relevant data for evaluation
                    datapoint = np.array([[dim, sparsity, seed, duration, iters, nx, ne, ni, true_sparsity, ip_converged, cvxpy_converged, duration_cvxpy, max_diff_cvxpy, mean_diff_cvxpy]])
                    
                    results = np.vstack((results, datapoint))
        
        # store performance on all instances of problem class per linear system solver and problem class.
        np.savetxt(f'{Path(__file__).parent}/data/{class_name}_{solver_name}_experiment_results.csv', results, delimiter=';')
