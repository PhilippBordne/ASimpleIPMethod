""" 
Some function to create plots for report.
"""


import numpy as np
from matplotlib import pyplot as plt

def plot_2d(Q: np.ndarray, c: np.ndarray, x_traj: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float,
            A: np.ndarray=None, b: np.ndarray=None, C: np.ndarray=None, d: np.ndarray=None):
    (X, Y) = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.stack((X, Y), 2)
    
    # plt contour lines of objective
    F = 0.5 * grid[:, :, None, :] @ (Q[None, None, :] @ grid[:, :, :, None]) + c[None, None, :] @ grid[:, :, :, None]
    F = F.squeeze()
    
    min_f = np.min(F)
    max_f = np.max(F)
    
    step_size = (max_f - min_f) / 10
    plt.contour(X, Y, F, [step_size * (i + 0.1) for i in range(10)])
    
    # if there is equality constraint plot it
    if A is not None:
        G = A[None, None, :]  @ grid[:, :, :, None] - b
        G = G.squeeze()
        plt.contour(X, Y, G, [0], colors='r')
    
    # if there are inequality constraint plot boundaries 
    if C is not None:
        H = C[None, None, :]  @ grid[:, :, :, None] - d[:, None]
        H = H[:, :, :, 0]
        for i in range(H.shape[-1]):
            plt.contour(X, Y, H[:, :, i], [0], colors='orange')

    # plot the trajectory
    plt.scatter(x_traj[0, -1], x_traj[1, -1], color='green', marker='o')
    plt.plot(x_traj[0, :], x_traj[1, :], color='black', marker='x', label="iterates x_k")
    plt.axis([x_min, x_max, y_min, y_max])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid()
    plt.legend()
    plt.show()


def plot_complementaritities():
    # (X, Y) = np.meshgrid(np.linspace(0, 5, 100), np.linspace(0, 5, 100))
    (X, Y) = np.meshgrid([i/100 for i in range(500)], [i/100 for i in range(500)])
    grid = np.stack((X, Y), axis=2)
    
    tau = 0.2
    
    non_smooth = X * Y
    smooth_1 = X * Y - tau
    smooth_2 = X * Y - 0.5 * tau
    smooth_3 = X * Y - 0.25 * tau
    smooth_4 = X * Y - 0.025 * tau
    
    fig, ax = plt.subplots()
    ax.axhline(y=0, color='k', linewidth=0.75)
    ax.axvline(x=0, color='k', linewidth=0.75)
    plt.plot([], [], color='#1f77b4', label=r"smoothened complementarities")
    plt.plot([], [], color='#ff7f0e', label="non-smooth complementarity")
    c1 = plt.contour(X, Y, smooth_1, [0], linewidths=2, colors='#1f77b4')
    c2 = plt.contour(X, Y, smooth_2, [0], linewidths=2, colors='#1f77b4')
    c3 = plt.contour(X, Y, smooth_3, [0], linewidths=2, colors='#1f77b4')
    c4 = plt.contour(X, Y, smooth_4, [0], linewidths=2, colors='#1f77b4')
    plt.contour(X, Y, non_smooth, [0], linewidths=3, colors='#ff7f0e')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.clabel(c1, inline=1, fontsize=10, inline_spacing=0, manual=[(0.75, 0)], fmt=r'$\tau=0.2$')
    plt.clabel(c2, inline=1, fontsize=10, inline_spacing=0, manual=[(0.75, 0)], fmt=r'$\tau=0.1$')
    plt.clabel(c3, inline=1, fontsize=10, inline_spacing=0, manual=[(0.75, 0)], fmt=r'$\tau=0.05$')
    plt.clabel(c4, inline=1, fontsize=10, inline_spacing=0, manual=[(0.75, 0)], fmt=r'$\tau=0.005$')
    # plt.title(r"smoothened complementarities for $\tau \in \{0.005, 0.05, 0.1, 0.2\}$")
    plt.xlim(-0.2, 1.5)
    plt.ylim(-0.2, 1.5)
    plt.xlabel(r'$\mu$')
    plt.ylabel(r's')
    plt.legend()
    plt.show()
    
    
# plot_complementaritities()
