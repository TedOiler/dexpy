import numpy as np
from tqdm import tqdm  # for progress bar
from scipy.optimize import minimize
from gen_rand_design import gen_rand_design  # custom function for generating random designs
from cordex_discrete import cordex_discrete


def cordex_continuous(runs, feats, J_cb=None, epochs=1000, method='Nelder-Mead', optimality='D', random_start=True, disable_bar=False):
    """
    Uses a coordinate descent algorithm to find the design with the minimum D-optimality (or maximum A-optimality)
    criterion value for a continuous regression problem.

    Args:
        runs (int): Number of runs/samples.
        feats (int): Number of features/parameters.
        J_cb (numpy.ndarray, optional): Matrix that represents the integral of basis multiplied together. Defaults to None.
        epochs (int, optional): Number of iterations. Defaults to 1000.
        method (str, optional): The optimization method to use. Should be one of the methods supported by
                                `scipy.optimize.minimize`. Defaults to 'Nelder-Mead'.
        optimality (str, optional): The optimality criterion to use. Should be one of 'D' or 'A', which correspond to
                                    the D-optimality and A-optimality criteria, respectively. Defaults to 'D'.

    Returns:
        tuple (numpy.ndarray, float): The design with the best D-optimality or A-optimality value, and the corresponding design.

    Raises:
        ValueError: If the specified optimality criterion is not one of 'D' or 'A'.
        ValueError: If the specified optimization method is not one of 'Nelder-Mead', 'Powell', 'TNC', or 'L-BFGS-B'.

    """

    # Objective function for D-optimality or A-optimality criterion
    def objective(x, optimality='A'):
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
        M = Zetta.T @ Zetta

        if optimality == 'D':
            try:
                cr = np.linalg.det(M)
            except np.linalg.LinAlgError:
                cr = np.infty
            return -cr  # negative of criterion value for minimization (i.e., maximize D-optimality)
        elif optimality == 'A':
            try:
                cr = np.trace(np.linalg.inv(M))  # trace of inverse of Zetta.T x Zetta
            except np.linalg.LinAlgError:
                cr = np.infty
            return cr
        elif optimality == 'E':
            try:
                cr = np.max(np.linalg.eigvals(M))
            except np.linalg.LinAlgError:
                cr = np.infty
            return cr
        elif optimality == "I":
            try:
                cr = np.min(np.linalg.eigvals(M))
            except np.linalg.LinAlgError:
                cr = np.infty
            return -cr
        else:
            raise ValueError(f"Invalid criterion {optimality}. "
                             "Criterion should be one of 'D', or 'A'.")

    if method not in ['Nelder-Mead', 'Powell', 'TNC', 'L-BFGS-B']:
        raise ValueError(f"Invalid method {method}. "
                         "Method should be one of 'Nealder-Mead', 'Powell', 'TNC', or 'L-BFGS-B'.")

    ones = np.array([1] * runs).reshape(-1, 1)  # create a column vector of ones with shape (runs, 1)
    epochs_list = []
    for epoch in tqdm(range(epochs), disable=disable_bar):
        if random_start:
            Gamma = gen_rand_design(runs=runs, feats=feats)
        else:
            Gamma, _ = cordex_discrete(runs=runs, feats=feats, levels=[-1, 0, 1], epochs=5, optimality=optimality, J_cb=J_cb, disable_bar=True)

        for run in range(runs):
            for feat in range(feats):
                x0 = Gamma[run, feat]  # set initial guess for optimization to current value of Gamma
                res = minimize(objective, x0, method=method, bounds=[(-1, 1)], args=optimality)
                Gamma[run, feat] = res.x
                cr = objective(res.x, optimality=optimality)
        epochs_list.append([epoch, cr, Gamma])
    epochs_list = np.array(epochs_list, dtype=object)
    Design_best = epochs_list[epochs_list[:, 1].argmin(), 2]
    Opt_best = epochs_list[epochs_list[:, 1].argmin(), 1]
    # Correct criterion sign
    Opt_best = -Opt_best if optimality in ['D', 'I'] else Opt_best
    return Design_best, Opt_best
