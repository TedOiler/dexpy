from .base_model import BaseModel
import numpy as np


class ScalarOnFunctionModel(BaseModel):
    def __init__(self, J_cb):
        self.J_cb = J_cb  # Matrix representing the integral of basis functions multiplied together

    def compute_objective(self, Model_mat, f_coeffs):
        ones = np.ones((Model_mat.shape[0], 1))
        Gamma = Model_mat[:, :f_coeffs]
        Zetta = np.concatenate((ones, Gamma @ self.J_cb), axis=1)
        Covar = Zetta.T @ Zetta

        try:
            P_inv = np.linalg.inv(Covar)
        except np.linalg.LinAlgError:
            return np.nan

        # A-optimality: Trace of the inverse of the information matrix
        value = np.trace(P_inv)

        # In practice, you might want to ensure 'value' is valid (e.g., not NaN or Inf) before returning
        return value
