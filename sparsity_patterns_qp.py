"""
Script to plot sparsity patterns of instances of investigated QP classes.
"""

from matplotlib import pyplot as plt
from matplotlib import gridspec
from qps import ControlQP, RandomQP

qp = ControlQP(10, seed=2)
# qp = RandomQP(100)
qp.get_x_sol_cvxpy()
Q = qp.Q
A = qp.A
C = qp.C
n = len(Q)
m_e = len(A)
m_i = len(C)

def plot_sparsities_matrices():
    marker_size = 4
    # Setting up the plot surface
    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(nrows=2, ncols=2, width_ratios=[1, 1], height_ratios=[n, m_e])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_title("Cost matrix $Q$", fontweight="bold")
    ax0.spy(Q, c='#1f77b4', markersize=marker_size)
    ax0.set_xticks([0, n])
    ax0.set_yticks([0, n])
    
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_title("System dynamics $A$", fontweight="bold")
    ax1.spy(A, c='#1f77b4', markersize=marker_size)# Third axes
    ax1.set_xticks([0, n])
    ax1.set_yticks([0, m_e])
    ax2 = fig.add_subplot(gs[:, 1])
    ax2.set_title("Path constraints $C$", fontweight="bold")
    ax2.spy(C, c='#1f77b4', markersize=marker_size)
    ax2.set_xticks([0, n])
    ax2.set_yticks([0, m_i])
    plt.show()

plot_sparsities_matrices()
