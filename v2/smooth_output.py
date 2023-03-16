import numpy as np
from basis import step


def smooth_fourier(nb, results):
    n = 100  # size of equidistant points
    h = 1 / n  # increment size
    T = np.reshape(np.arange(h, 1, h), (-1, 1))
    f = [step(t, w=results) for t in T]

    base = np.zeros((nb - 1, n - 1))
    sins_idx = np.reshape(2 * (np.arange(0, np.floor(nb / 2), 1, dtype=int)), (-1, 1))
    coss_idx = np.reshape(2 * (np.arange(1, np.ceil(nb / 2), 1, dtype=int)) - 1, (-1, 1))
    base[np.reshape(sins_idx, (-1,))] = np.sqrt(2) * np.sin((np.pi * sins_idx) @ T.T)
    base[np.reshape(coss_idx, (-1,))] = np.sqrt(2) * np.cos((np.pi * (coss_idx - 1)) @ T.T)
    base[1] = base[1] / np.sqrt(2)  # Orthonormalization
    Fbase = base[1:, :].T

    Cf = np.linalg.inv(Fbase.T @ Fbase) @ Fbase.T @ f  # OLS
    recon_f = Fbase @ Cf
    return recon_f.reshape(-1, 1).T
