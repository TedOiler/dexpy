import numpy as np
from matplotlib import pyplot as plt

from basis import step

plt.style.use('fivethirtyeight')
# plt.style.use('default')
# plt.style.use('bmh')
# plt.style.use('ggplot')
# plt.style.use('grayscale')


def plot_basis(ax, T, w, f) -> None:
    """
    Plots a step function with knots and weights.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot the function.
        T (array_like): The array of points to plot the step function on.
        w (array_like): The array of weights for each knot.
        f (function): The function used to generate the step function.

    Returns:
        None
    """

    ax.plot(T, [f(t, w=w) for t in T], zorder=-1)
    ax.set_ylim(-1.2, 1.2)
    # Calculate knots and weights
    knots = [(1 / (len(w) - 1 + 1)) * (i + 1) for i in
             range(len(w) - 1 + 1 - 1)]
    weights = [w[i + 1] for i in range(
        len(w) - 1)]  # we want to exclude the first and last points, as these will be drawn with a different colour

    # Draw knots
    ax.scatter(knots, weights, color="darkorange", s=35, zorder=1)  # internal knots
    ax.scatter([0, 1], [w[0], w[len(w) - 1]], color="black", s=35, zorder=1)  # support knots
    # ax.set_xlabel("$t$")
    # ax.set_ylabel("$f(t)$")
    # ax.grid(visible=False)
    ax.locator_params(axis='y', nbins=3)
    ax.locator_params(axis='x', nbins=6)


def subplot_results(sub_x, sub_y, T, results, J_cb) -> None:
    """
    Plots multiple subplots of step functions with knots and weights.

    Args:
        sub_x (int): The number of rows of subplots.
        sub_y (int): The number of columns of subplots.
        T (array_like): The array of points to plot the step functions on.
        results (array_like): The array of weight vectors for each step function.
        J_cb (array_like): The array of weights for the control points.

    Returns:
        None
    """
    fig, ax = plt.subplots(sub_x, sub_y)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
    # fig.tight_layout()
    row_to_plot = 0
    for i in range(sub_x):
        for j in range(sub_y):
            plot_basis(ax=ax[i, j], T=T, w=results[row_to_plot, :].tolist(), f=step)
            row_to_plot += 1
    Z = np.concatenate((np.array([1] * (sub_x * sub_y)).reshape(-1, 1), results @ J_cb), axis=1)
    opt = np.round(np.trace(np.linalg.inv(Z.T @ Z)), 3)
    # fig.suptitle(f'A-opt: {opt}')
    plt.show()
    # print(f'The results plotted are: \n {results}')
    # print(f'With corresponding matrix Z (= 1 | results @ J_cb):\n {Z}')
    # print(f'With information matrix M (= Z.T @ Z):\n {Z.T @ Z}')
    # print(f'Inverted $M^{{-1}}$:\n {np.linalg.inv(Z.T @ Z)}')
    # print(f'With determinat of the inverse of the information matrix {np.linalg.det(np.linalg.inv(Z.T @ Z))}\n')
    # print(f'And A-optimality criterion A (tr[$M^{{-1}}$]): {np.trace(np.linalg.inv(Z.T @ Z))}')
