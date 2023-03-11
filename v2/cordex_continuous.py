import numpy as np
from tqdm import tqdm  # for progress bar
from scipy.optimize import minimize
from gen_rand_design import gen_rand_design  # custom function for generating random designs
from cordex_discrete import cordex_discrete


def find_best_design(epochs_list, optimality):
    # filter only the positive objectives (bug)
    if optimality in ['A']:
        epochs_list = epochs_list[epochs_list[:, 1] > 0]
    else:
        epochs_list = epochs_list[epochs_list[:, 1] < 0]
    min_objective = epochs_list[:, 1].argmin()
    Design_best = epochs_list[min_objective, 2]
    Opt_best = epochs_list[min_objective, 1]
    # Correct criterion sign
    Opt_best = -Opt_best if optimality in ['D'] else Opt_best
    return Design_best, Opt_best, epochs_list


def cordex_continuous(runs, feats, num_f, num_x, J_cb=None, R_0=None, epochs=1000, method='Nelder-Mead', optimality='A',
                      random_start=True, disable_bar=True, penalty=0):
    """
    Uses a coordinate descent algorithm to find the design with the minimum D-optimality (or maximum A-optimality)
    criterion value for a continuous regression problem.

    Args:
        runs (int): Number of runs/samples.
        feats (int): Number of features/parameters.
        J_cb (numpy.ndarray, optional): Matrix that represents the integral of basis multiplied together. Defaults to None.
        R_0 (numpy.ndarray, optional): Matrix of m-th order smoothness penalty. Defaults to None.
        epochs (int, optional): Number of iterations. Defaults to 1000.
        method (str, optional): The optimization method to use. Should be one of the methods supported by
                                `scipy.optimize.minimize`. Defaults to 'Nelder-Mead'.
        optimality (str, optional): The optimality criterion to use. Should be one of 'D' or 'A', which correspond to
                                    the D-optimality and A-optimality criteria, respectively. Defaults to 'D'.
        random_start (bool, optional): If set to False, the starting design will be selected by running the discrete
                                        coordinate exchange. This will make the algorithm run slower, but produce better results
                                        Defaults to True.
        disable_bar (bool, optional): This will set the tqdm progress bar of the internal algorithm to False.
                                        Defaults to True
        penalty (int, optional): The lambda parameter for the penalization matrix. Defaults to 0.

    Returns:
        tuple (numpy.ndarray, float): The design with the best D-optimality or A-optimality value, and the corresponding design.

    Raises:
        ValueError: If the specified optimality criterion is not one of 'D' or 'A'.
        ValueError: If the specified optimization method is not one of 'Nelder-Mead', 'Powell', 'TNC', or 'L-BFGS-B'.

    """

    def objective(x):
        """
        Objective function for D-optimality or A-optimality criterion.
        Args:
            x (numpy.ndarray): The coordinate that is to be changed in the design matrix.
            optimality (str, optional): The optimality criterion to use. Should be one of 'D' or 'A', which correspond
                                        to the D-optimality and A-optimality criteria, respectively. Defaults to 'A'.
        Returns:
            float: The D-optimality or A-optimality value of the design.
        Raises:
            ValueError: If the specified optimality criterion is not one of 'D' or 'A'.
        """
        Gamma[run, feat] = x
        Zetta = np.concatenate((ones, Gamma), axis=1) if J_cb is None else np.concatenate((ones, Gamma @ J_cb), axis=1)
        Mu = Zetta.T @ Zetta

        if R_0 is None:
            MATRIX = Mu
        else:
            try:
                P_inv = np.linalg.inv(Mu + penalty * R_0)
            except np.linalg.LinAlgError:
                P_inv = None
            MATRIX = P_inv @ Mu @ P_inv

        if optimality == 'D':
            try:
                criterion_value = np.linalg.det(MATRIX)
            except np.linalg.LinAlgError:
                criterion_value = np.infty
            return -criterion_value  # negative of criterion value for minimization (i.e., maximize D-optimality)
        elif optimality == 'A':
            try:
                criterion_value = np.trace(np.linalg.inv(MATRIX))  # trace of inverse of Zetta.T x Zetta
            except np.linalg.LinAlgError:
                criterion_value = np.infty
            return criterion_value
        else:
            raise ValueError(f"Invalid criterion {optimality}. "
                             "Criterion should be one of 'D', or 'A'.")

    if method not in ['Nelder-Mead', 'Powell', 'TNC', 'L-BFGS-B']:
        raise ValueError(f"Invalid method {method}. "
                         "Method should be one of 'Nelder-Mead', 'Powell', 'TNC', or 'L-BFGS-B'.")
    ones = np.ones((runs, 1))
    epochs_list = []
    for epoch in tqdm(range(epochs), disable=not disable_bar):
        if random_start:
            Gamma = gen_rand_design(runs=runs, feats=feats)
        else:
            Gamma, _, _ = cordex_discrete(runs=runs, feats=feats, levels=[-1, 0, 1], epochs=10, optimality=optimality,
                                          J_cb=J_cb, disable_bar=disable_bar)
        for run in range(runs):
            for feat in range(feats):
                x0 = Gamma[run, feat]  # set initial guess for optimization to current value of Gamma
                res = minimize(objective, x0, method=method, bounds=[(-1, 1)])
                Gamma[run, feat] = res.x
                objective_value = objective(res.x)
        epochs_list.append([epoch, objective_value, Gamma])

    return find_best_design(np.array(epochs_list, dtype=object), optimality)
