import numpy as np


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
