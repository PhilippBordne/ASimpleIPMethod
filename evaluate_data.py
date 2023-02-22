import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd

(IDX_DIM, IDX_SPARSE, IDX_SEED, IDX_DUR, IDX_ITER, IDX_NX, IDX_NE, IDX_NI, IDX_T_SPARSE, IDX_IP_CONV, IDX_CVXPY_CONV, IDX_CVXPY_DUR,
IDX_MAX_DIFF, IDX_MEAN_DIFF) = np.arange(0, 14)


path_to_data = Path(__file__).parent / 'data'

# load experimental data into pyton data frame
data_LDLT_qp = pd.read_csv(f"{path_to_data}/random_qp_LDLT_experiment_results.csv", delimiter=';', header=None)
data_LDLT_control = pd.read_csv(f"{path_to_data}/control_LDLT_experiment_results.csv", delimiter=';', header=None)
data_LDLT_own_qp = pd.read_csv(f"{path_to_data}/random_qp_LDLT_own_experiment_results.csv", delimiter=';', header=None)
data_LDLT_own_control = pd.read_csv(f"{path_to_data}/control_LDLT_own_experiment_results.csv", delimiter=';', header=None)
data_LU_qp = pd.read_csv(f"{path_to_data}/random_qp_LU_experiment_results.csv", delimiter=';', header=None)
data_LU_control = pd.read_csv(f"{path_to_data}/control_LU_experiment_results.csv", delimiter=';', header=None)


# # plot runtime for the qp benchmarks for all sparsities settings over the dimension in x
# data_ldlt = data_LDLT_qp.groupby([IDX_DIM])[IDX_DUR].mean()
# data_lu = data_LU_qp.groupby([IDX_DIM])[IDX_DUR].mean()
# data_cvxpy = data_LDLT_qp.loc[data_LDLT_qp[IDX_CVXPY_CONV] != 0.0].groupby([IDX_DIM])[IDX_CVXPY_DUR].mean()
# plt.plot(data_ldlt, label="LDLT")
# plt.plot(data_lu, label="LU")
# plt.plot(data_cvxpy, label="CVXPY")
# plt.xlabel("Dimension decision variable x")
# plt.ylabel("Runtime [s]")
# plt.yscale('log')
# plt.legend()
# plt.show()

# plot runtime for the qp benchmarks for decision dim 50 over the sparsities
# data_ldlt = data_LDLT_qp.loc[data_LDLT_qp[IDX_DIM] == 50].groupby([IDX_T_SPARSE])[IDX_DUR].mean()
# data_lu = data_LU_qp.loc[data_LU_qp[IDX_DIM] == 50].groupby([IDX_T_SPARSE])[IDX_DUR].mean()
# data_cvxpy = data_LDLT_qp.loc[(data_LDLT_qp[IDX_DIM] == 50) & (data_LDLT_qp[IDX_CVXPY_CONV] == 1)].groupby([IDX_T_SPARSE])[IDX_CVXPY_DUR].mean()

data_ldlt = data_LDLT_qp.loc[data_LDLT_qp[IDX_DIM] == 50]

plt.scatter(data_ldlt[IDX_T_SPARSE], data_ldlt[IDX_DUR], label="LDLT")
# plt.plot(data_ldlt, label="LDLT")
# plt.plot(data_lu, label="LU")
# plt.plot(data_cvxpy, label="CVXPY")
plt.xlabel("Sparsity parameter")
plt.ylabel("Runtime [s]")
# plt.yscale('log')
plt.legend()
plt.show()


# avg_dur_over_dim_qp_ldlt = data_LDLT_qp.groupby([IDX_DIM])[IDX_DUR].mean()
# # avg_dur_over_dim_qp_ldlt = avg_dur_over_dim_qp_ldlt.reset_index()
# # print(avg_dur_over_dim_qp_ldlt)
# # print(avg_dur_over_dim_qp_ldlt.loc[avg_dur_over_dim_qp_ldlt[IDX_DIM] == 10.0])

# plt.plot(avg_dur_over_dim_qp_ldlt)
# plt.show()


# # plt.scatter(res_ldlt_dim, res_ldlt_dur)

# # # plt.scatter(res_ldlt[:, 0], res_ldlt[:, 1], label="LDLT")
# # # plt.scatter(res_lu[:, 0], res_lu[:, 1], label="LU")
# # plt.legend()
# # plt.show()
