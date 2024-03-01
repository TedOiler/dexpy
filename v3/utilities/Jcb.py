import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag

from .basis import indicator, polynomial


def elements(n, p, l1=0) -> float:
    """
    Calculates the integral of a polynomial basis function over a subinterval of [0,1] with length 1/n.
    The subinterval is determined by the value of l1.

    Args:
        n (int): The total number of subintervals.
        p (int): The degree of the polynomial to integrate.
        l1 (float, optional): The starting point of the subinterval. Defaults to 0.

    Returns:
        float: The value of the integral.
    """
    return quad(lambda t: (indicator(t, l=l1 / n, u=(l1 + 1) / n)) * (polynomial(t, p=p)), l1 / n, (l1 + 1) / n,
                full_output=True)[0]


def calc_basis_matrix(x_basis, b_basis) -> np.ndarray:
    """
    Calculates the matrix of basis functions for the input dataset.

    Parameters:
        x_basis (int): The number of subintervals to divide the input data into.
        b_basis (int): The degree of the polynomial basis functions to use.

    Returns:
        np.ndarray: A matrix of size (x_basis x b_basis) containing the values of the polynomial basis
        functions evaluated over the subintervals.
    """
    return np.array([[elements(n=x_basis, p=p, l1=l1) for p in range(b_basis)] for l1 in range(x_basis)])


def Jcb(*matrices):
    return block_diag(*matrices)
