"""
Script used to obtain sparsities of KKT systems for RandomQP instances generated using different sparsitz
parameters rho.
"""

from qps import RandomQP, ControlQP
import numpy as np

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
