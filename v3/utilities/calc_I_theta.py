import numpy as np


def calc_I_theta(Kx, Ky):
    return np.eye((len(Kx)+1)*Ky, dtype=float)