import numpy as np


def gen_rand_design(runs: int, feats: int, low: int = -1, high: int = 1) -> np.ndarray:
    """
    Generates a random `runs` x `feats` matrix with each element uniformly distributed between -1 and 1.

    Args:
        runs (int): The number of rows in the matrix.
        feats (int): The number of columns in the matrix.

    Returns:
        np.ndarray: A random `runs` x `feats` matrix with each element uniformly distributed between -1 and 1.
    """
    A = np.random.uniform(low=low, high=high, size=(runs, feats))
    # while np.linalg.det(A.T @ A) == 0:
    #     A = np.random.rand(runs, feats) - 0.5
    return A
