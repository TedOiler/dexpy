from .base_optimizer import BaseOptimizer
from scipy.optimize import minimize
import numpy as np
import sys
from pathlib import Path

current_dir = Path(__file__).parent.resolve()
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))
from utilities.gen_rand_design import gen_rand_design_m
from models.f_on_f import FunctionOnFunctionModel
from tqdm import tqdm


class CordexContinuous(BaseOptimizer):
    def __init__(self, model):
        super().__init__(model)

    def optimize(self, runs, nx, scalars=0, epochs=1000, final_pass_iter=100):
        best_design = None
        best_objective_value = np.inf

        for _ in tqdm(range(epochs)):
            Gamma_, X_ = gen_rand_design_m(runs=runs, f_list=nx, scalars=scalars)
            DesignMatrix = Gamma_
            for i in range(DesignMatrix.shape[0]):
                for j in range(DesignMatrix.shape[1]):
                    if isinstance(self.model, FunctionOnFunctionModel):
                        # Adjust the objective function call for FunctionOnFunctionModel
                        objective = lambda x: self.model.compute_objective(DesignMatrix, runs, nx)
                    else:
                        # Call for ScalarOnFunctionModel remains as before
                        objective = lambda x: self.model.compute_objective(DesignMatrix, sum(nx) + scalars)
                    result = minimize(objective, DesignMatrix[i, j], method='L-BFGS-B', bounds=[(-1, 1)])
                    if result.x is not None:
                        DesignMatrix[i, j] = result.x
                    current_value = objective(result.x)

            if 0 <= current_value < best_objective_value:
                best_objective_value = current_value
                best_design = DesignMatrix

        if final_pass_iter > 0:
            for _ in tqdm(range(final_pass_iter)):
                current_value = best_objective_value
                for i in range(DesignMatrix.shape[0]):
                    for j in range(DesignMatrix.shape[1]):
                        result = minimize(objective, DesignMatrix[i, j], method='L-BFGS-B', bounds=[(-1, 1)])
                        if result.x is not None:
                            DesignMatrix[i, j] = result.x
                        current_value = objective(result.x)
                if 0 <= current_value < best_objective_value:
                    best_objective_value = current_value
                    best_design = DesignMatrix

        return best_design, np.abs(best_objective_value)
