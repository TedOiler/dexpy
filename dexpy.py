import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures  # TODO: implement what I need from this package


class DesignParams:
    pass


class Design:
    """
    Class Docstring.
    """

    def __init__(self, experiments=None, levels=None):
        """
        :param int experiments: Number of Experiments to design
        :param dict levels: Levels of factors
        Constructor Docstring.
        """

        self.experiments = experiments
        self.levels = levels
        self.features = len(levels.keys())

        self.order = None
        self.interactions_only = None
        self.bias = None

        self.epochs = None
        self.criterion = None

    # ---------------DUNDER, GETTERS AND SETTERS FUNCTION---------------------------------------------------------------
    def __repr__(self):
        return f"Design(experiments={self.experiments}, levels={self.levels})"

    def set_model(self, order, interactions_only=False, bias=True):
        """
        :param int order: Order of the polynomial (1-main effects, 2-quadratic effects, ...)
        :param bool interactions_only: Include terms as x1^2 or not
        :param bool bias: Include a beta_0 on the design matrix or not

        Setter for model parameters
        """
        self.order = order
        self.interactions_only = interactions_only
        self.bias = bias

    def set_algorithm(self, epochs, criterion):
        """
        :param int epochs: Number of random start to check
        :param str criterion: What criterion to use for maximization. Includes ("A", "C", "D", "E", "S", "T", "G", "I", "V")

        Setter for algorithm parameters
        """
        self.epochs = epochs
        self.criterion = criterion

    # ------------------------------------------------------------------------------------------------------------------

    def gen_random_design(self) -> pd.DataFrame:
        """
        Generate a random starting design matrix.
        """
        df = pd.DataFrame(np.random.random((self.experiments, self.features)))
        df.columns = ['x' + str(x) for x in list(range(self.features))]
        return df

    def gen_model_matrix(self, data=None) -> pd.DataFrame:
        """
        :param pd.DataFrame data: Design matrix
        Generate the model matrix of a design matrix (argument)
        """
        if any(var is None for var in [self.order, self.interactions_only, self.bias]):
            raise Exception('Parameters: \'order\', \'interactions_only\' and \'bias\' cannot be None')

        poly = PolynomialFeatures(degree=self.order,
                                  interaction_only=self.interactions_only,
                                  include_bias=self.bias)
        df = pd.DataFrame(poly.fit_transform(data))
        df.columns = poly.get_feature_names(data.columns)
        return df

    @staticmethod
    def clear_histories(design_matrix, epoch, all_opt_cr, max_bool=True):
        """
       :param pd.DataFrame design_matrix: Number of Experiments to design
       :param int epoch: Number of random start to check
       :param list all_opt_cr: Levels of factors
       :param bool max_bool: Should the criterion be maximized (True) or minimizes (False)?

       Run the coordinate exchange algorithm and produce the best model matrix, according to the criterion chosen, as well as a history of all other possible model matrices and the history of the selected criterion used.
       """
        designs, histories = pd.DataFrame(), pd.DataFrame()
        if 'epoch' not in design_matrix:
            design_matrix['epoch'] = epoch
        history = pd.DataFrame(all_opt_cr)
        if max_bool:
            history['max'] = history.iloc[:, 2:].max(axis=1)
        else:
            history['min'] = history.iloc[:, 2:].min(axis=1)

        if 'epoch' not in history:
            history['epoch'] = epoch
        d = designs.append(design_matrix, ignore_index=True)
        h = histories.append(history, ignore_index=True)
        return d, h

    @staticmethod
    def find_best_design(histories, designs, max_bool=True):
        """
        :param pd.DataFrame histories: Dataframe of all the histories per epoch
        :param pd.DataFrame designs: Dataframe of all the designs per epoch
        :param bool max_bool: Should the criterion be maximized (True) or minimizes (False)?

        Group the histories per epoch and getting the max. Then, the function uses that max index (best epoch) to retrieve the design of that epoch and save it as the best design.
        The function also changes behaviour according to the max_bool flag which is used to tell the function if we are searching for a maximum of a minimum.
        """
        if max_bool:
            per_epoch = histories.groupby('epoch')['max'].max()
            return designs[designs['epoch'] == per_epoch.idxmax()].reset_index().iloc[:, 1:-1]
        else:
            per_epoch = histories.groupby('epoch')['min'].min()
            return designs[designs['epoch'] == per_epoch.idxmin()].reset_index().iloc[:, 1:-1]

    @staticmethod
    def guards():
        pass

    def fit(self):
        self.guards()

        all_opt_cr = []  # data for each level

        for epoch in range(self.epochs):
            design_matrix = self.gen_random_design()
            for exp in range(self.experiments):
                for feat in range(self.features):
                    coordinate_opt_cr = []
                    for count, level in enumerate(self.levels[feat]):
                        # check all possible levels for the specific experiment, feature
                        design_matrix.iat[exp, feat] = level
                        model_matrix = self.gen_model_matrix(data=design_matrix)
                        det = np.linalg.det(model_matrix.T @ model_matrix)
                        coordinate_opt_cr.append(det)

                    all_opt_cr.append([exp, feat, *coordinate_opt_cr])
                    # updated design_matrix
                    design_matrix.iat[exp, feat] = self.levels[feat][coordinate_opt_cr.index(max(coordinate_opt_cr))]


            # clean results of inner loops
            designs, histories = self.clear_histories(design_matrix=design_matrix, epoch=epoch, all_opt_cr=all_opt_cr)

        best_design = self.find_best_design(histories=histories, designs=designs)
        model_matrix = self.gen_model_matrix(data=best_design)

        return model_matrix, best_design, designs, histories
