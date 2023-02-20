import numpy as np
from tqdm import tqdm
import torch
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.acquisition.monte_carlo import qUpperConfidenceBound
from botorch.acquisition.monte_carlo import qProbabilityOfImprovement
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.settings import suppress_botorch_warnings
import warnings
from gen_rand_design import gen_rand_design
from cordex_continuous import cordex_continuous
from cordex_discrete import cordex_discrete

suppress_botorch_warnings(True)
warnings.filterwarnings('ignore', category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tkwargs = {'dtype': torch.double, 'device': device}


def gen_next_point(X, y, best_y, n_exp, acq_f='UCB', inequality_constraints=None):
    """
    Generates the next set of candidates for Bayesian optimization using the acquisition function specified.

    Args:
        X (torch.Tensor): Tensor containing the input features of the training data.
        y (torch.Tensor): Tensor containing the output values of the training data.
        best_y (float): The best output value observed so far.
        n_exp (int): The number of candidates to generate.
        acq_f (str, optional): The acquisition function to use. Can be one of 'EI', 'PI', or 'UCB'. Defaults to 'UCB'.
        inequality_constraints (tuple, optional): A tuple of callables defining the inequality constraints. Defaults to None.

    Returns:
        torch.Tensor: A tensor of shape (n_exp, X.shape[1]) containing the new candidates.

    Raises:
        ValueError: If the specified acquisition function is not valid.
    """
    bounds = torch.Tensor([[-1] * X.shape[1], [1] * X.shape[1]], **tkwargs)

    model = SingleTaskGP(X, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    if acq_f == 'EI':
        acq_f = qExpectedImprovement(model=model, best_f=best_y)
    elif acq_f == 'NEI':
        acq_f = qNoisyExpectedImprovement(model=model, X_baseline=X)
    elif acq_f == 'PI':
        acq_f = qProbabilityOfImprovement(model=model, best_f=best_y)
    elif acq_f == 'UCB':
        acq_f = qUpperConfidenceBound(model=model, beta=1e-8)
    else:
        raise ValueError(f"Invalid acquisition function {acq_f}. "
                         "acq_f should be one of 'EI', 'PI', or 'UCB'.")

    candidates, _ = optimize_acqf(acq_function=acq_f,
                                  bounds=bounds,
                                  q=n_exp,
                                  num_restarts=100,
                                  raw_samples=2000,
                                  options={"batch_limit": 5, "maxiter": 200},
                                  inequality_constraints=inequality_constraints)
    return candidates


def bo_loop(epochs, runs, feats, optimality, J_cb, n_exp=1, acq_f='EI', initialization_method=None,
            inequality_constraint=None):
    """
    Run a Bayesian optimization loop.

    Args: epochs (int): The number of optimization iterations to run. runs (int): The number of experimental runs.
    feats (int): The number of features in the design. optimality (str): The optimality criterion to use. Should be
    one of 'A', 'D', 'E' or 'I'. J_cb (Optional[np.ndarray]): If provided, a matrix to use in transforming the design
    before evaluating it. n_exp (int): The number of new candidate points to generate at each iteration. acq_f (str):
    The acquisition function to use. Should be one of 'EI', 'PI', or 'UCB'. initialization_method (str): The method
    used on the continuous coordinate exchange to give the bo_loop a good starting design. Should be one of
    'Nelder-Mead', 'L-BFGS-B', 'TNC', or 'Powell' inequality_constraint (Optional[dict]): An optional dictionary of
    inequality constraints to apply to the design.

    Returns:
        Tuple[np.ndarray, float]: The best design found, and the corresponding best objective function value.
    """

    def objective(X, optimality=optimality):
        """
        Computes the objective function value for a given design.

        Args:
            X : np.ndarray of shape (n_samples, n_features)
                Design matrix.
            optimality : str, default='A'
                The criterion used to compute the optimality. It can be one of the following: 'D' (Determinant), 'A' (Average Diagonal), 'E' (Maximum Eigenvalue), and 'I' (Minimum Eigenvalue).

        Returns:
            np.ndarray of shape (1,)
                Objective function value.

        Raises:
            ValueError
                If the optimality criterion is not one of 'D', 'A', 'E', or 'I'.
        """
        ones = np.array([1] * runs).reshape(-1, 1)
        X = X.reshape(runs, feats)
        Zetta = np.concatenate((ones, X), axis=1) if J_cb is None else np.concatenate((ones, X @ J_cb), axis=1)
        M = Zetta.T @ Zetta

        if optimality == 'D':
            try:
                cr = np.linalg.det(M)
            except np.linalg.LinAlgError:
                cr = np.infty
            return np.array([cr])
        elif optimality == 'A':
            try:
                cr = np.trace(np.linalg.inv(M))
            except np.linalg.LinAlgError:
                cr = np.infty
            return np.array([-cr])
        elif optimality == 'E':
            try:
                cr = np.max(np.linalg.eigvals(M))
            except np.linalg.LinAlgError:
                cr = np.infty
            return np.array([-cr])
        elif optimality == 'I':
            try:
                cr = np.min(np.linalg.eigvals(M))
            except np.linalg.LinAlgError:
                cr = np.infty
            return np.array([cr])

        else:
            raise ValueError(f"Invalid criterion {optimality}. "
                             "Criterion should be one of 'D', 'A', 'E', or 'I'.")

    def get_initial_data_cordex_cont():
        """
        Generates initial data points for Bayesian optimization using the continuous Cordex algorithm.

        Returns a tuple with the following elements:
        - X: tensor of shape (1, runs * feats) containing the flattened matrix of initial design points
        - y: tensor of shape (runs, 1) containing the objective function values for each initial design point in X
        - best_y: float representing the highest objective function value in y

        The Cordex algorithm is used to generate the initial design points. The number of design points is specified
        by the 'runs' and 'feats' parameters. The dataset is generated using the 'cordex_continuous' function from
        the 'cordex_continuous' module, which generates a dataset with continuous design points. The design points
        are flattened into a 1D tensor X of shape (1, runs * feats).

        The objective function values for each design point are obtained by calling the 'objective' function,
        which takes X as input and returns a tensor of objective function values y of shape (runs, 1).

        The highest objective function value in y is returned as the 'best_y' value.
        """
        X, _, _ = cordex_continuous(runs=runs,
                                    feats=J_cb.shape[0],
                                    epochs=3,  # Epochs could also be ceil(J_cb.shape[0]/10)
                                    method=initialization_method,
                                    J_cb=J_cb,
                                    optimality=optimality)
        X_flat = X.flatten().reshape(1, runs * feats)
        y = objective(X_flat, optimality=optimality)
        best_y = y.max().item()
        return torch.tensor(X_flat), torch.tensor(y).reshape(-1, 1), best_y

    def get_initial_data_cordex_disc():
        X, best_cr, _ = cordex_discrete(runs=runs,
                                        feats=J_cb.shape[0],
                                        epochs=5,  # Epochs could also be ceil(J_cb.shape[0]/10)
                                        optimality=optimality,
                                        levels=[-1, 1],
                                        J_cb=J_cb)
        X_flat = X.flatten().reshape(1, runs * feats)
        y = objective(X_flat, optimality=optimality)
        if optimality in ['A', 'E']:
            best_y = -best_cr
        else:
            best_y = best_cr
        return torch.tensor(X_flat), torch.tensor(y).reshape(-1, 1), best_y

    def get_initial_data():
        """
        Generates initial data for Bayesian optimization by creating a random design matrix X and evaluating the objective
        function for each row of X. Returns the flattened design matrix X, the corresponding objective function values y,
        and the best objective function value found in y.

        Args:

        Returns:
            Tuple[torch.Tensor, torch.Tensor, float]: A tuple containing the flattened design matrix X, the objective
            function values y, and the best objective function value found in y.
        """
        X = gen_rand_design(runs=runs, feats=feats)
        X_flat = X.flatten().reshape(1, runs * feats)
        y = objective(X_flat, optimality=optimality)
        best_y = y.max().item()
        return torch.tensor(X_flat), torch.tensor(y).reshape(-1, 1), best_y

    # Choose which function will generate the data according to user input
    if initialization_method == 'Discrete':
        initialization_function = get_initial_data_cordex_disc
    elif initialization_method is not None:
        initialization_function = get_initial_data_cordex_cont
    else:
        initialization_function = get_initial_data

    X_init, y_init, best_y_init = initialization_function()
    epochs_list = [[-1, best_y_init, X_init]]
    for epoch in tqdm(range(epochs)):
        try:
            new_candidates = gen_next_point(X=X_init, y=y_init, best_y=best_y_init, n_exp=n_exp, acq_f=acq_f,
                                            inequality_constraints=inequality_constraint)
        except:
            # delete last row since it is the one that cased the problem
            X_init = X_init[:-1, :]
            y_init = y_init[:-1, :]
            continue
        new_results = objective(X=new_candidates, optimality=optimality)
        new_results = torch.Tensor(new_results.reshape(-1, 1))
        X_init = torch.cat([X_init, new_candidates])
        y_init = torch.cat([y_init, new_results])
        best_y_init = y_init.max().item()
        epochs_list.append([epoch, new_results.numpy()[0][0], new_candidates])

    epochs_list = np.array(epochs_list, dtype=object)
    epochs_max_id = epochs_list[:, 1].argmax()
    Best_des = epochs_list[epochs_max_id, 2]
    Opt_best = epochs_list[epochs_max_id, 1]
    # Correct criterion sign
    Opt_best = -Opt_best if optimality in ['A', 'E'] else Opt_best
    return np.array(Best_des).reshape(runs, feats), Opt_best, epochs_list
