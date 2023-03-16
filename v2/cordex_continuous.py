import numpy as np
from tqdm import tqdm  # for progress bar
from scipy.optimize import minimize
from gen_rand_design import gen_rand_design_m  # custom function for generating random designs
from cordex_discrete import cordex_discrete


def cordex_continuous(runs, f_list, scalars, optimality='A', J_cb=None, R_0=None, smooth_pen=0, ridge_pen=0,
                      epochs=1000,
                      method='L-BFGS-B', random_start=False, disable_bar=True, final_pass=False):
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
        disable_bar (bool, optional): This will set the tqdm progress bar of the internal algorithm to False.
                                        Defaults to True.
        final_pass (bool, optional): If set to True, the algorithm will make a final pass after completion to make sure that the
                                        final design is the true best. This is useful when running a lot of designs. Defaults to False.

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

        def MATRIX_calculation():
            if R_0 is None and ridge_pen == 0:
                P = np.identity(Mu.shape[0])
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
            return P_inv @ Mu @ P_inv

        Model_mat[run, feat] = x
        Gamma = Model_mat[:, :f_coeffs]
        X = Model_mat[:, f_coeffs:]
        Zetta = np.concatenate((ones, Gamma @ J_cb, X), axis=1)
        Mu = Zetta.T @ Zetta

        MATRIX = MATRIX_calculation()

        # if R_0 is None:
        #     MATRIX = Mu
        # else:
        #     try:
        #         P_inv = np.linalg.inv(Mu + penalty * R_0)
        #     except np.linalg.LinAlgError:
        #         P_inv = None  # not good enough need to handle that differently
        #     MATRIX = P_inv @ Mu @ P_inv
        if optimality == 'D':
            try:
                value = np.linalg.det(MATRIX)
            except np.linalg.LinAlgError:
                value = np.nan
            return -value  # negative of criterion value for minimization (i.e., maximize D-optimality)
        elif optimality == 'A':
            try:
                value = np.trace(np.linalg.inv(MATRIX))  # trace of inverse of Zetta.T x Zetta
            except np.linalg.LinAlgError:
                value = np.nan
            return value
        else:
            raise ValueError(f"Invalid criterion {optimality}. "
                             "Criterion should be one of 'D', or 'A'.")

    def check(obj):
        if optimality in ['A']:
            return 0 < obj < Best_obj
        elif optimality in ['D']:
            return obj < 0 < Best_obj

    if method not in ['Nelder-Mead', 'Powell', 'TNC', 'L-BFGS-B']:
        raise ValueError(f"Invalid method {method}. "
                         "Method should be one of 'Nelder-Mead', 'Powell', 'TNC', or 'L-BFGS-B'.")
    if runs < len(f_list) + scalars:
        raise ValueError(f"Design not Estimable."
                         f"Runs {runs}, Parameters: {sum(f_list) + scalars}")

    Best_des = None
    Best_obj = np.inf
    f_coeffs = sum(f_list) + 1
    ones = np.ones((runs, 1))

    for _ in tqdm(range(epochs), disable=not disable_bar):
        if random_start:
            Gamma_, X_ = gen_rand_design_m(runs=runs, f_list=f_list, scalars=scalars)
            Model_mat = np.hstack((Gamma_, X_))
        else:
            Model_mat, _ = cordex_discrete(runs=runs, f_list=f_list, scalars=scalars, levels=[-1, 1], epochs=10,
                                           optimality=optimality, J_cb=J_cb, disable_bar=disable_bar)
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
        for run in range(Best_des.shape[0]):
            for feat in range(Best_des.shape[1]):
                x0 = Best_des[run, feat]
                res = minimize(objective, x0, method=method, bounds=[(-1, 1)])
                Best_des[run, feat] = res.x
                objective_value = objective(res.x)
        if check(objective_value):
            Best_obj = objective_value
            Best_des = Model_mat

    return Best_des, np.abs(Best_obj)
