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
from models.s_on_f import ScalarOnFunctionModel
from tqdm import tqdm


class CordexContinuous(BaseOptimizer):
    def __init__(self, model):
        super().__init__(model)

    def optimize(self, runs, nx, scalars=0, epochs=1000, final_pass_iter=100):

        if isinstance(self.model, FunctionOnFunctionModel):
            objective = lambda x: self.model.compute_objective_input(x, i, j, model_matrix, runs, nx)
        elif isinstance(self.model, ScalarOnFunctionModel):
            objective = lambda x: self.model.compute_objective_input(x, i, j, model_matrix, sum(nx) + scalars)

        best_design = None
        best_optimality_value = np.inf

        for _ in tqdm(range(epochs)):
            Gamma_, X_ = gen_rand_design_m(runs=runs, f_list=nx, scalars=scalars)
            model_matrix = Gamma_
            for i in range(model_matrix.shape[0]):
                for j in range(model_matrix.shape[1]):
                    result = minimize(objective, model_matrix[i, j], method='L-BFGS-B', bounds=[(-1, 1)])
                    if result.x is not None:
                        model_matrix[i, j] = result.x
                    current_optimality_value = objective(result.x)

            if 0 <= current_optimality_value < best_optimality_value:
                best_optimality_value = current_optimality_value
                best_design = model_matrix

        if final_pass_iter > 0:
            for _ in tqdm(range(final_pass_iter)):
                current_optimality_value = best_optimality_value
                for i in range(model_matrix.shape[0]):
                    for j in range(model_matrix.shape[1]):
                        result = minimize(objective, model_matrix[i, j], method='L-BFGS-B', bounds=[(-1, 1)])
                        if result.x is not None:
                            model_matrix[i, j] = result.x
                        current_optimality_value = objective(result.x)
                if 0 <= current_optimality_value < best_optimality_value:
                    best_optimality_value = current_optimality_value
                    best_design = model_matrix

        return best_design, np.abs(best_optimality_value)
