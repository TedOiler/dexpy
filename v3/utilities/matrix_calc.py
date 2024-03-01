import numpy as np
from scipy.integrate import quad
from scipy.linalg import block_diag

from .basis import indicator, polynomial


def elements(n, p, l1=0) -> float:
    return quad(lambda t: (indicator(t, l=l1 / n, u=(l1 + 1) / n)) * (polynomial(t, p=p)), l1 / n, (l1 + 1) / n,
                full_output=True)[0]


def calc_basis_matrix(x_basis, b_basis) -> np.ndarray:
    return np.array([[elements(n=x_basis, p=p, l1=l1) for p in range(b_basis)] for l1 in range(x_basis)])


def Jcb(*matrices):
    return block_diag(*matrices)


def calc_Sigma(Kx, Ky, N, decay=0):
    In = np.eye(N)
    inputs = np.linspace(0, 1, (len(Kx) + 1) * Ky)

    def exp_decay(dec_rate, x):
        return np.exp(-dec_rate * x)

    if decay == 0:
        sigma1 = np.diag(exp_decay(decay, inputs))
    elif decay == np.inf:
        elements = np.zeros((len(Kx) + 1) * Ky)
        # elements[0] = 1 # not sure
        sigma1 = np.diag(elements)
        sigma1[0, 0] = 1
    else:
        sigma1 = np.diag(exp_decay(decay, inputs))
    return np.kron(In, sigma1)


def calc_J_CH(Kx, Kb):
    J_chs = [Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(x_row, b_row)]) for x_row, b_row in zip(Kx, Kb)]
    bases = [1] + J_chs
    return block_diag(*bases)


def calc_I_theta(Kx, Ky):
    return np.eye((len(Kx)+1)*Ky, dtype=float)