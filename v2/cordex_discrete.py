import numpy as np
from tqdm import tqdm
from gen_rand_design import gen_rand_design
from gen_rand_design import gen_rand_design_m


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

def find_best_design(epochs_list, optimality):
    epochs_list = epochs_list[~np.isnan(epochs_list[:, 1].astype(float))]
    Design_best = epochs_list[epochs_list[:, 1].argmin(), 2]
    Opt_best = epochs_list[epochs_list[:, 1].argmin(), 1]
    # Correct criterion sign
    Opt_best = -Opt_best if optimality in ['D', 'I'] else Opt_best
    return Design_best, Opt_best, epochs_list


def cordex_discrete(runs, f_list, scalars, levels, epochs, optimality='A', J_cb=None, disable_bar=False) -> object:
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

    def objective():
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
        Gamma = Model_mat[:, :f_coeffs]
        X = Model_mat[:, f_coeffs:]
        Zetta = np.hstack((ones, Gamma @ J_cb, X))
        M = Zetta.T @ Zetta

        if optimality == "D":
            try:
                value = np.linalg.det(M)
            except np.linalg.LinAlgError:
                value = np.nan
            return -value
        elif optimality == "A":
            try:

                value = np.trace(np.linalg.inv(M))
            except np.linalg.LinAlgError:
                value = np.nan
            return value
        else:
            raise ValueError(f"Invalid criterion {optimality}. "
                             "Criterion should be one of 'D', 'A', 'E', or 'I'.")

    f_coeffs = sum(f_list)+1
    ones = np.ones((runs, 1))
    epochs_list = []

    for epoch in tqdm(range(epochs), disable=disable_bar):
        Gamma_, X_ = gen_rand_design_m(runs=runs, f_list=f_list, scalars=scalars)  # [n x n_x]
        Model_mat = np.hstack((Gamma_, X_))
        for run in range(runs):
            for feat in range(f_coeffs + scalars - 1):
                best_level_list = []
                for level in levels:
                    Model_mat[run, feat] = level
                    objective_value = objective()
                    best_level_list.append(objective_value)
                best_level_index = best_level_list.index(min(best_level_list))
                Model_mat[run, feat] = levels[best_level_index]

        # For each epoch, compute the optimality criterion to keep in an array.
        objective_value = objective()
        epochs_list.append([epoch, objective_value, Model_mat])
    return find_best_design(np.array(epochs_list, dtype=object), optimality)
