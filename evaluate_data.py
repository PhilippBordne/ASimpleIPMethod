import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

(IDX_DIM, IDX_SPARSE, IDX_SEED, IDX_DUR, IDX_ITER, IDX_NX, IDX_NE, IDX_NI, IDX_T_SPARSE, IDX_IP_CONV, IDX_CVXPY_CONV, IDX_CVXPY_DUR,
IDX_MAX_DIFF, IDX_MEAN_DIFF) = np.arange(0, 14)


path_to_data = Path(__file__).parent / 'data'

# load experimental data into pandas data frame
def load_data(problem_class: str):
    data_ldlt_eigen = pd.read_csv(f"{path_to_data}/{problem_class}_LDLT_experiment_results.csv", delimiter=';', header=None)
    data_ldlt_scipy = pd.read_csv(f"{path_to_data}/{problem_class}_LDLT_own_experiment_results.csv", delimiter=';', header=None)
    data_lu = pd.read_csv(f"{path_to_data}/{problem_class}_LU_experiment_results.csv", delimiter=';', header=None)
    return data_ldlt_eigen, data_ldlt_scipy, data_lu


# plot runtime for the qp benchmarks for all sparsities settings over the dimension in x
def plot_runtimes(data_ldlt_eigen, data_ldlt_scipy, data_lu, data_cvxpy, criterion: str, log=True, xticks=None):
    plt.plot(data_ldlt_eigen, label="LDLT (Eigen)")
    plt.plot(data_ldlt_scipy, label="LDLT (scipy)")
    plt.plot(data_lu, label="LU")
    plt.plot(data_cvxpy, label="CVXPY")
    if criterion == "n":
        plt.xlabel(r"$n$")
    elif criterion == "rho":
        plt.xlabel(r"$\rho$")
    plt.ylabel(r"Runtime $[s]$")
    if log:
        plt.yscale('log')
    if xticks is not None:
        plt.xticks(xticks)
    plt.grid()
    plt.legend()
    plt.show()

 
def plot_runtimes_overall(problem_class: str, log=True):
    data_ldlt_eigen, data_ldlt_scipy, data_lu = load_data(problem_class)
    data_ldlt_eigen = data_ldlt_eigen.groupby([IDX_DIM])[IDX_DUR].mean()
    data_ldlt_scipy = data_ldlt_scipy.groupby([IDX_DIM])[IDX_DUR].mean()
    data_cvxpy = data_lu.loc[data_lu[IDX_CVXPY_CONV] != 0.0].groupby([IDX_DIM])[IDX_CVXPY_DUR].mean()
    data_lu = data_lu.groupby([IDX_DIM])[IDX_DUR].mean()
    
    plot_runtimes(data_ldlt_eigen, data_ldlt_scipy, data_lu, data_cvxpy, criterion="n", log=log)
    
def plot_runtimes_sparsities(dim):
    data_ldlt_eigen, data_ldlt_scipy, data_lu = load_data("random_qp")
    data_ldlt_eigen = data_ldlt_eigen.loc[data_ldlt_eigen[IDX_DIM] == dim].groupby([IDX_SPARSE])[IDX_DUR].mean()
    data_ldlt_scipy = data_ldlt_scipy.loc[data_ldlt_scipy[IDX_DIM] == dim].groupby([IDX_SPARSE])[IDX_DUR].mean()
    data_cvxpy = data_lu.loc[(data_lu[IDX_DIM] == dim) & (data_lu[IDX_CVXPY_CONV] == 1)].groupby([IDX_SPARSE])[IDX_CVXPY_DUR].mean()
    data_lu = data_lu.loc[data_lu[IDX_DIM] == dim].groupby([IDX_SPARSE])[IDX_DUR].mean()
    plot_runtimes(data_ldlt_eigen, data_ldlt_scipy, data_lu, data_cvxpy, "rho", log=False, xticks=[0.08, 0.09, 0.1, 0.15, 0.3])
    
def ratio_of_problems_solved(problem_class):
    data_ldlt_eigen, data_ldlt_scipy, data_lu = load_data(problem_class)
    print(f"Ratio of problems solved LDLT Eigen:\n {data_ldlt_eigen[[IDX_IP_CONV, IDX_CVXPY_CONV]].aggregate(['mean'])}")
    print(f"Mean difference x solutions CVXPY - LDLT Eigen:\n {data_ldlt_eigen[IDX_MAX_DIFF].aggregate(['mean'])}")
    print(f"Max difference x solutions CVXPY - LDLT Eigen:\n {data_ldlt_eigen[IDX_MAX_DIFF].aggregate(['max'])}")
    print(f"Ratio of problems solved LDLT scipy:\n {data_ldlt_scipy[[IDX_IP_CONV, IDX_CVXPY_CONV]].aggregate(['mean'])}")
    print(f"Mean difference x solutions CVXPY - LDLT scipy:\n {data_ldlt_scipy[IDX_MAX_DIFF].aggregate(['mean'])}")
    print(f"Max difference x solutions CVXPY - LDLT scipy:\n {data_ldlt_scipy[IDX_MAX_DIFF].aggregate(['max'])}")
    print(f"Ratio of problems solved LU:\n {data_lu[[IDX_IP_CONV, IDX_CVXPY_CONV]].aggregate(['mean'])}")
    print(f"Mean difference x solutions CVXPY - LU:\n {data_lu[IDX_MAX_DIFF].aggregate(['mean'])}")
    print(f"Max difference x solutions CVXPY - LU:\n {data_lu[IDX_MAX_DIFF].aggregate(['max'])}")

# ratio_of_problems_solved("random_qp")
# ratio_of_problems_solved("control")

plot_runtimes_overall("random_qp", log=False)
