import numpy as np
from skopt import gp_minimize
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import RBF
from skopt.optimizer import Optimizer
from skopt.space import Real
from tqdm import tqdm

from skopt.plots import plot_objective
from skopt.plots import plot_evaluations
from matplotlib import pyplot as plt
import seaborn as sns


def objective_function(X, m, n, J_cb=None, noise=0):
    ones = np.ones((m, 1)).reshape(-1, 1)
    X = np.array(X).reshape(m, n)
    Z = np.hstack((ones, X @ J_cb))
    try:
        M = np.linalg.inv(Z.T @ Z)
    except np.linalg.LinAlgError:
        return 1e10
    result = np.trace(M) + np.random.normal(0, noise)
    if result < 0:
        return 1e10
    else:
        return result


def objective_function_latent(latent_X, m, n, J_cb, decoder, noise=0):
    X_next = decoder.predict(np.array(latent_X).reshape(1, -1))
    X_next = X_next.flatten()
    return objective_function(X_next, m, n, J_cb=J_cb, noise=noise)


def latent_auto_bo(runs, n_x, J_cb, decoder, latent_dim, n_iterations=50, n_random_starts=5, random_seed=42, n_runs=1,
                   noise=0, acq_func="LCB", initial_point_generator="random"):
    all_results = []
    search_space = [Real(-1., 1.) for _ in range(latent_dim)]
    for run in range(n_runs):

        gp_result = gp_minimize(
            lambda x: objective_function_latent(x, runs, sum(n_x), J_cb, decoder, noise=noise),
            search_space,
            n_calls=n_iterations,
            n_random_starts=n_random_starts,
            random_state=random_seed + run,
            verbose=True,
            acq_func=acq_func,
            initial_point_generator=initial_point_generator,
        )

        optimal_latent_X_gp = gp_result.x
        optimal_y_gp = gp_result.fun
        optimal_X_gp = decoder.predict(np.array(optimal_latent_X_gp).reshape(1, -1))
        optimal_X_gp = optimal_X_gp.flatten()

        optimal_matrix_gp = np.array(optimal_X_gp).reshape(runs, sum(n_x))
        optimal_det_gp = optimal_y_gp

        all_results.append((optimal_matrix_gp, optimal_det_gp, gp_result))

    return all_results


def latent_manual_bo(runs, n_x, J_cb, decoder, latent_dim, n_iterations=200, random_seed=42):
    # Adjust the search space to match the latent space dimension
    search_space = [Real(-1., 1.) for _ in range(latent_dim)]

    # Create a Gaussian Process (GP) model with a Radial Basis Function (RBF) kernel
    kernel = RBF(length_scale=1, length_scale_bounds=(1e-2, 1e2))
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, random_state=random_seed)
    opt = Optimizer(search_space, base_estimator=gpr, n_initial_points=5, acq_func="LCB", random_state=random_seed)

    def damped_oscillation(x):
        return A * np.exp(-b * x) * (1 + np.sin((2 * np.pi) / c * x)) + minimum

    A = 0.5  # starting position
    b = 0.02  # decay rate
    c = 10  # period
    minimum = 1e-2  # minimum value
    for i in tqdm(range(n_iterations)):
        kappa = damped_oscillation(i)
        opt.acq_func_kwargs = {'kappa': kappa}

        latent_X = opt.ask()
        f_val = objective_function_latent(latent_X, runs, sum(n_x), J_cb, decoder, noise=0)
        opt.tell(latent_X, f_val)

    # Find the optimal point in the latent space
    yi_array = np.array(opt.yi)
    positive_yi_indices = np.where(yi_array > 0)
    min_positive_yi_index = positive_yi_indices[0][np.argmin(yi_array[positive_yi_indices])]
    optimal_latent_X = opt.Xi[min_positive_yi_index]

    # Decode the optimal point in the latent space back into the design matrix space
    optimal_X = decoder.predict(np.array(optimal_latent_X).reshape(1, -1))
    optimal_X = optimal_X.flatten()

    optimal_matrix = np.array(optimal_X).reshape(runs, sum(n_x))
    optimal_det = np.min(yi_array[positive_yi_indices])

    return optimal_matrix, optimal_det


def plot_convergence(results, title, threshold=None):
    plt.style.use('default')
    plt.figure(figsize=(8, 6))

    lines = []
    labels = []

    # Generate a color palette with the same number of colors as the runs
    colors = sns.color_palette("husl", n_colors=len(results))

    # Calculate minimum values for all runs
    all_min_values = [np.nanmin(np.minimum.accumulate(result[-1].func_vals)) for result in results]

    # Find the index of the best run
    best_run_idx = np.argmin(all_min_values)

    for idx, result in enumerate(results):
        min_values = np.minimum.accumulate(result[-1].func_vals)  # Calculate the running minimum value

        if threshold is not None:
            min_values = np.where(min_values < threshold, min_values, np.nan)  # Replace extreme values with NaNs

        min_value = np.nanmin(min_values)  # Calculate the minimum value ignoring NaNs

        # Assign a bright red color to the best run and duller colors to the other runs
        color = "red" if idx == best_run_idx else tuple(0.5 * c for c in colors[idx])

        line, = plt.plot(range(1, len(result[-1].func_vals) + 1), min_values, marker="o", color=color)

        lines.append(line)
        labels.append((f"Run {idx+1}", min_value))

    # Sort the labels based on the minimum values
    labels.sort(key=lambda x: x[1])
    sorted_labels = [label[0] for label in labels]

    plt.xlabel("Number of calls")
    plt.ylabel("Minimum objective function value")
    plt.title(title)
    plt.legend(lines, sorted_labels)
    plt.grid()
    plt.show()


def plot_evals(gp_result):
    plt.style.use('default')
    _ = plot_evaluations(gp_result)
    plt.show()


def plot_obj(gp_result):
    plt.style.use('default')
    _ = plot_objective(gp_result)
    plt.show()
