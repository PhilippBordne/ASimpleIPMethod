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

# def function_2d(Q: np.ndarray, x: np.ndarray):

# plot_2d(np.eye(2), np.array([-1, -1]), np.array([1, -1]), np.array([-1]), np.array([[2, -1], [4, -1]]), np.array([-1, -1]),
        # -4, 4, -4, 4)
