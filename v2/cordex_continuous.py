import numpy as np
from tqdm import tqdm  # for progress bar
from scipy.optimize import minimize
from gen_rand_design import gen_rand_design_m  # custom function for generating random designs
from cordex_discrete import cordex_discrete


def cordex_continuous(runs, f_list, scalars, optimality='A', J_cb=None, R_0=None, smooth_pen=0, ridge_pen=0,
                      epochs=1000, estimator='MLE', smoothness_lambda=0,
                      method='L-BFGS-B', random_start=False, final_pass=True, final_pass_iter=100,
                      main_bar=True, starting_bar=False, final_bar=False):
    """
    Uses a coordinate descent algorithm to find the design with the minimum D-optimality (or maximum A-optimality)
    criterion value for a continuous regression problem.

    Args:
        runs (int): Number of runs/samples.
        f_list (list): Number of features/parameters.
        scalars (int): Number of scalar components.
        optimality (str, optional): The optimality criterion to use. Should be one of 'D' or 'A', which correspond to
                                    the D-optimality and A-optimality criteria, respectively. Defaults to 'A'.
        J_cb (numpy.ndarray, optional): Matrix that represents the integral of basis multiplied together. Defaults to None.
        R_0 (numpy.ndarray, optional): Matrix of m-th order smoothness penalty. Defaults to None.
        smooth_pen (int, optional): The lambda parameter for the penalization matrix. Defaults to 0.
        ridge_pen (int, optional): Ridge penalty for numerical stability. Defaults to 0.
        epochs (int, optional): Number of iterations. Defaults to 1000.
        method (str, optional): The optimization method to use. Should be one of the methods supported by
                                `scipy.optimize.minimize`. Defaults to 'L-BFGS-B'.
        random_start (bool, optional): If set to False, the starting design will be selected by running the discrete
                                        coordinate exchange. This will make the algorithm run slower, but produce better results
                                        Defaults to False.
        final_pass (bool, optional): If set to True, the algorithm will make a final pass after completion to make sure that the
                                        final design is the true best. This is useful when running a lot of designs. Defaults to False.
        final_pass_iter (int, optional): Number of iterations for the final pass. Defaults to 100.
        main_bar (bool, optional): This will set the tqdm progress bar of the main algorithm.
                                        Defaults to True.
        starting_bar (bool, optional): This will set the tqdm progress bar of the starting design.
                                        Defaults to False.
        final_bar (bool, optional): This will set the tqdm progress bar of the final pass.
                                        Defaults to False.
    Returns:
        tuple (numpy.ndarray, float): The design with the best D-optimality or A-optimality value, and the corresponding design.

    Raises:
        ValueError: If the specified optimality criterion is not one of 'D' or 'A'.
        ValueError: If the specified optimization method is not one of 'Nelder-Mead', 'Powell', 'TNC', or 'L-BFGS-B'.
        ValueError: If the number of runs is less than the number of features plus the number of scalar components.

    """

    def objective(x):
        """
        Objective function for D-optimality or A-optimality criterion.

        Args:
            x (numpy.ndarray): The coordinate that is to be changed in the design matrix.

        Returns:
            float: The D-optimality or A-optimality value of the design.

        Raises:
            ValueError: If the specified optimality criterion is not one of 'D' or 'A'.
        """

        Model_mat[run, feat] = x
        Gamma = Model_mat[:, :f_coeffs]
        X = Model_mat[:, f_coeffs:]
        Zetta = np.concatenate((ones, Gamma @ J_cb, X), axis=1)
        Mu = Zetta.T @ Zetta

        # Good idea but it needs a PEA instead of a CEA to work properly
        # smoothness_penalty = 0
        # for r in range(Model_mat.shape[0]):
        #     for i in range(1, Model_mat.shape[1]):
        #         smoothness_penalty += np.abs(Model_mat[r, i] - Model_mat[r, i - 1])

        if R_0 is None and ridge_pen == 0:
            P = Mu
        else:
            if R_0 is None:
                P = Mu + ridge_pen * np.identity(Mu.shape[0])
            elif ridge_pen == 0:
                P = Mu + smooth_pen * R_0
            else:
                P = Mu + smooth_pen * R_0 + ridge_pen * np.identity(Mu.shape[0])
        try:
            P_inv = np.linalg.inv(P)
        except np.linalg.LinAlgError:
            return np.nan

        if estimator == 'MLE':
            MATRIX = P_inv @ Mu @ P_inv
        elif estimator == 'Bayes':
            MATRIX = P_inv
        else:
            raise ValueError(f"Invalid estimator {estimator}. "
                             "Estimator should be one of 'MLE', or 'Bayes'.")
        if np.isclose(np.linalg.det(MATRIX), 0):
            return np.nan

        if optimality == 'D':
            value = np.linalg.det(MATRIX)
            if np.isclose(value, 0, atol=1e-8):
                value = np.nan
            return value
        elif optimality == 'A':
            value = np.trace(MATRIX)
            if np.isclose(value, 0, atol=1e-8):
                value = np.nan
            return value
        else:
            raise ValueError(f"Invalid criterion {optimality}. "
                             "Criterion should be one of 'D', or 'A'.")

    def check(obj):
        if optimality in ['A']:
            return 0 <= obj < Best_obj
            # return obj < Best_obj
        elif optimality in ['D']:
            return 0 <= obj < Best_obj
            # return obj < Best_obj

    def run_checks():
        if method not in ['Nelder-Mead', 'Powell', 'TNC', 'L-BFGS-B']:
            raise ValueError(f"Invalid method {method}. "
                             "Method should be one of 'Nelder-Mead', 'Powell', 'TNC', or 'L-BFGS-B'.")
        if runs < J_cb.shape[1] + scalars + 1 and smooth_pen == 0 and ridge_pen == 0:
            raise ValueError(f"Design not Estimable."
                             f"Runs {runs}, Parameters: {sum(f_list) + scalars}, with no penalty.")
        if R_0 is None and smooth_pen != 0:
            raise ValueError(f"Smoothness penalty is set to {smooth_pen}, but no smoothness matrix is provided.")
        if R_0 is not None and smooth_pen == 0:
            raise ValueError(f"Smoothness matrix is provided, but smoothness penalty is set to {smooth_pen}.")

    run_checks()
    Best_des = None
    Best_obj = np.inf
    f_coeffs = sum(f_list) + 1
    ones = np.ones((runs, 1))
    Model_mat = None

    for _ in tqdm(range(epochs), disable=not main_bar):
        objective_value = np.inf
        if random_start:
            Gamma_, X_ = gen_rand_design_m(runs=runs, f_list=f_list, scalars=scalars)
            Model_mat = np.hstack((Gamma_, X_))
        else:
            Model_mat, _ = cordex_discrete(runs=runs, f_list=f_list, scalars=scalars, levels=[-1, 1], epochs=10,
                                           optimality=optimality, J_cb=J_cb, disable_bar=not starting_bar)
        for run in range(runs):
            for feat in range(f_coeffs + scalars - 1):
                res = minimize(objective, Model_mat[run, feat], method=method, bounds=[(-1, 1)])
                if res.x is not None:
                    Model_mat[run, feat] = res.x
                objective_value = objective(res.x)

        if check(objective_value):
            Best_obj = objective_value
            Best_des = Model_mat

    if final_pass:
        if final_bar:
            print("Executing final pass...")
        for _ in tqdm(range(final_pass_iter), disable=not final_bar):
            objective_value = Best_obj
            for run in range(Model_mat.shape[0]):
                for feat in range(Model_mat.shape[1]):
                    res = minimize(objective, Model_mat[run, feat], method=method, bounds=[(-1, 1)])
                    Model_mat[run, feat] = res.x
                    objective_value = objective(res.x)
                    if np.isclose(objective_value, 0, atol=1e-15):
                        objective_value = 0
            if check(objective_value):
                Best_obj = objective_value
                Best_des = Model_mat

    return Best_des, np.abs(Best_obj)
