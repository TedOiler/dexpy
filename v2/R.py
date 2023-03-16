import numpy as np
from scipy.misc import derivative
from scipy.integrate import quad
from basis import polynomial
from scipy.linalg import block_diag


def calc_penalty_matrix(n_b, padding=True):
    zeros_ver = np.zeros(n_b).reshape(-1, 1)
    zeros_hor = np.zeros(n_b + 1).reshape(1, -1)

    R_mat = np.array(
        [[quad(lambda t: (derivative(polynomial, t, n=2, dx=1e-6, args=[j])) *
                         (derivative(polynomial, t, n=2, dx=1e-6, args=[i])),
               0, 1, full_output=True)[0] for j in range(n_b)] for i in range(n_b)])

    if padding:
        R_0 = np.round(np.concatenate((zeros_hor, np.concatenate((zeros_ver, R_mat), axis=1)), axis=0), 3)
    else:
        R_0 = np.round(R_mat, 3)
    return R_0


def R(*matrices):
    return block_diag(*matrices)
