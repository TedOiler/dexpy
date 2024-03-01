from .base_model import BaseModel
import numpy as np


class FunctionOnFunctionModel(BaseModel):
    def __init__(self, I_theta, J_CH, Sigma):
        self.I_theta = I_theta  # Identity matrix scaled by the size of Kx and Ky
        self.J_CH = J_CH  # Integral of basis matrix from calc_J_CH
        self.Sigma = Sigma  # Error structure matrix from calc_Sigma

    def compute_objective(self, Gamma_, N, Kx):
        Gamma = np.hstack((np.ones((N, 1)), Gamma_))
        Z = Gamma @ self.J_CH
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        L1 = ZtZ_inv @ Z.T
        L = np.kron(self.I_theta, L1)
        Covar = L @ self.Sigma @ L.T

        # A-optimality: Trace of the covariance matrix
        value = np.trace(Covar)

        # In practice, you might want to ensure 'value' is valid (e.g., not NaN or Inf) before returning
        return value if np.isfinite(value) else np.nan
