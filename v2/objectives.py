import numpy as np


def objective_min(x, Gamma, run, feat, J_cb=None, R_0=None, optimality='A', penalty=0):
    ones = np.ones((Gamma.shape[0], 1))
    Gamma[run, feat] = x
    Zetta = np.hstach((ones, Gamma)) if J_cb is None else np.hstack((ones, Gamma @ J_cb))
    Mu = Zetta.T @ Zetta

    if R_0 is None:
        MATRIX = Mu
    else:
        try:
            P_inv = np.linalg.inv(Mu + penalty * R_0)
        except np.linalg.LinAlgError:
            P_inv = None
        MATRIX = P_inv @ Mu @ P_inv

    # Variance of estimator with penalty term:
    # var(b) = (Z.T @ Z + λ*R_0)^{-1} Z.T @ Z * * (Z.T @ Z + λ*R_0)^{-1} =
    #        = P_inv @ M @ P_inv

    if optimality == 'A':
        try:
            objective = np.trace(np.linalg.inv(MATRIX))
        except np.linalg.LinAlgError:
            objective = np.infty
        objective = 1. * objective
    elif optimality == 'D':
        try:
            objective = np.linalg.det(MATRIX)
        except np.linalg.LinAlgError:
            objective = np.infty
        objective = -1. * objective
    else:
        raise ValueError(f"Invalid criterion {optimality}. "
                         "Criterion should be one of 'D', or 'A'.")
    return objective


def objective_max(x, Gamma, run, feat, J_cb=None, optimality='A', pen_temp=0):
    return -objective_min(x=x, Gamma=Gamma, run=run, feat=feat, J_cb=J_cb, optimality=optimality, penalty=pen_temp)
