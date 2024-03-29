from .base_model import BaseModel
import numpy as np


class ScalarOnFunctionModel(BaseModel):
    def __init__(self, J_cb):
        self.J_cb = J_cb

    def compute_objective(self, Model_mat, f_coeffs):
        ones = np.ones((Model_mat.shape[0], 1))
        Gamma = Model_mat[:, :f_coeffs]
        Zetta = np.concatenate((ones, Gamma @ self.J_cb), axis=1)
        Covar = Zetta.T @ Zetta

        try:
            P_inv = np.linalg.inv(Covar)
        except np.linalg.LinAlgError:
            return np.nan

        value = np.trace(P_inv)

        return value

    def compute_objective_input(self, x, i, j, Model_mat, f_coeffs):
        Model_mat[i, j] = x
        return self.compute_objective(Model_mat, f_coeffs)
