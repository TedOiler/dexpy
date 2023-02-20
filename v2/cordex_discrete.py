import numpy as np
from tqdm import tqdm
from gen_rand_design import gen_rand_design


# The `cordex` function is a function for generating a design matrix for a simulation experiment. The function takes
# in several arguments:
#
# `runs`: an integer representing the number of runs of the simulation
# `feats`: an integer representing the number of features or variables in the simulation
# `levels`: a list of numbers representing the possible levels that the variables in the simulation can take on
# `epochs`: an integer representing the number of iterations to generate the design matrix
# `J_cb`: an optional parameter, a matrix of coefficients used in the calculation of the A-opt criterion
#
# The function begins by creating a matrix of ones with the same number of rows as the number of runs, and a single
# column. It then enters a loop for the number of specified epochs. Within this loop, a random design matrix `Gamma`
# is generated with the same number of rows as runs and the same number of columns as features.
#
# For each element in the `Gamma` matrix, the function calculates the A-opt criterion for each level in the `levels`
# list, and finds the minimum value of the A-opt criterion. The element in the `Gamma` matrix is then reset to the
# corresponding level from the `levels` list.
#
# After all the elements in the `Gamma` matrix have been updated, the A-opt criterion is calculated using the updated
# matrix, and the results are appended to an array.
#
# After all the epochs have been completed, the function finds the minimum A-opt value and returns the corresponding
# design matrix, as well as the minimum A-opt value.
#
# This function uses the `gen_rand_design` function to generate a random design matrix, `np.concatenate` to
# concatenate the matrix of ones with the matrix of coefficients, `np.linalg.inv` to calculate the inverse of a
# matrix, `np.trace` to calculate the trace of a matrix, `np.array` to create an array and `tqdm` which is a library
# used to show the progress of a loop.


def cordex_discrete(runs, feats, levels, epochs, optimality='A', J_cb=None, disable_bar=False) -> object:
    """
    Generates a discrete design matrix that optimizes a given criterion.

    Args:
        runs (int): Number of experimental runs.
        feats (int): Number of features for each run.
        levels (list): List of discrete levels to use for each element of the design matrix.
        epochs (int): Number of times to run the optimization algorithm.
        optimality (str, optional): The optimization criterion to use. Default is 'A'.
            Possible values are:
            - 'D': maximize determinant of the design matrix.
            - 'A': minimize the trace of the inverse of the product of the transpose of the design matrix and itself.
            - 'E': maximize the maximum eigenvalue of the design matrix.
            - 'I': minimize the minimum eigenvalue of the design matrix.
        J_cb (np.ndarray, optional): Matrix of integral of each base multiplied together. Default is None.

    Returns:
        tuple: A tuple containing the discrete design matrix that optimizes the criterion, and the value of the
            criterion.

    Raises:
        ValueError: If `optimality` is not one of 'D', 'A', 'E', or 'I'.
    """

    def objective(Gamma, optimality="D"):
        """
        This function calculates the optimality of a given matrix M based on the specified criterion. The available criteria are 'D' (Determinant), 'A' (Average Diagonal), 'E' (Maximum Eigenvalue), and 'I' (Minimum Eigenvalue).

        Args:
            M (numpy.ndarray): The input matrix for which the optimality will be computed.
            optimality (str): The criterion used to compute the optimality. It can be one of the following: 'D' (Determinant), 'A' (Average Diagonal), 'E' (Maximum Eigenvalue), and 'I' (Minimum Eigenvalue). Default is 'D'.

        Returns:
            cr (float): The optimality value of the input matrix based on the specified criterion.

        Raises:
            ValueError: If an invalid criterion is provided. The criterion should be one of 'D', 'A', 'E', or 'I'.
            np.linalg.LinAlgError: If an error occurs during the computation of the criterion. In this case, the optimality value is set to infinity.
        """
        Zetta = np.concatenate((ones, Gamma), axis=1) if J_cb is None else np.concatenate((ones, Gamma @ J_cb), axis=1)
        M = Zetta.T @ Zetta

        if optimality == "D":
            try:
                cr = np.linalg.det(M)
            except np.linalg.LinAlgError:
                cr = np.infty
            return -cr
        elif optimality == "A":
            try:
                cr = np.trace(np.linalg.inv(M))
            except np.linalg.LinAlgError:
                cr = np.infty
            return cr
        elif optimality == "E":
            try:
                cr = np.max(np.linalg.eigvals(M))
            except np.linalg.LinAlgError:
                cr = np.infty
            return cr
        elif optimality == "I":
            try:
                cr = np.min(np.linalg.eigvals(M))
            except np.linalg.LinAlgError:
                cr = np.infty
            return -cr
        else:
            raise ValueError(f"Invalid criterion {optimality}. "
                             "Criterion should be one of 'D', 'A', 'E', or 'I'.")

    ones = np.array([1] * runs).reshape(-1, 1)  # [n x 1]
    epochs_list = []
    for epoch in tqdm(range(epochs), disable=disable_bar):
        Gamma = gen_rand_design(runs=runs, feats=feats)  # [n x n_x]
        for run in range(runs):
            for feat in range(feats):
                best_level_list = []
                for level in levels:
                    Gamma[run, feat] = level
                    cr = objective(Gamma=Gamma, optimality=optimality)
                    best_level_list.append(cr)
                best_level_index = best_level_list.index(min(best_level_list))
                Gamma[run, feat] = levels[best_level_index]

        # For each epoch, compute the optimality criterion to keep in an array.
        cr = objective(Gamma=Gamma, optimality=optimality)
        epochs_list.append([epoch, cr, Gamma])
    epochs_list = np.array(epochs_list, dtype=object)
    Design_best = epochs_list[epochs_list[:, 1].argmin(), 2]
    Opt_best = epochs_list[epochs_list[:, 1].argmin(), 1]
    # Correct criterion sign
    Opt_best = -Opt_best if optimality in ['D', 'I'] else Opt_best
    return Design_best, Opt_best, epochs_list
