from .Jcb import Jcb, calc_basis_matrix
from scipy.linalg import block_diag


def calc_J_CH(Kx, Kb):
    J_chs = [Jcb(*[calc_basis_matrix(x_basis=x, b_basis=b) for x, b in zip(x_row, b_row)]) for x_row, b_row in zip(Kx, Kb)]
    bases = [1] + J_chs
    return block_diag(*bases)
