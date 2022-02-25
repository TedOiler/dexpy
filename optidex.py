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
    def clear_histories(optimalities, designs, design_mat):
        """
       :param list designs: Number of Experiments to design
       :param list optimalities: Number of random start to check
       :param pd.DataFrame design_mat: Should the criterion be maximized (True) or minimizes (False)?

       Run the coordinate exchange algorithm and produce the best model matrix, according to the criterion chosen, as well as a history of all other possible model matrices and the history of the selected criterion used.
       """

        hstry_designs = pd.DataFrame(designs, columns=['epoch', *list(design_mat.columns)])
        hstry_opt_cr = pd.DataFrame(optimalities).rename(columns={0: 'epoch',
                                                                  1: 'experiment',
                                                                  2: 'feature'})
        hstry_opt_cr['max'] = hstry_opt_cr.iloc[:, 3:].max(axis=1)

        return hstry_designs, hstry_opt_cr

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
            return designs[designs['epoch'] == per_epoch.idxmax()].reset_index().iloc[:, 2:]
        else:
            per_epoch = histories.groupby('epoch')['min'].min()
            return designs[designs['epoch'] == per_epoch.idxmin()].reset_index().iloc[:, 2:]

    @staticmethod
    def guards():
        pass

    @staticmethod
    def d_opt(matrix):
        return np.linalg.det(matrix.T @ matrix)

    def fit(self):
        self.guards()

        hstry_opt_cr = []  # all optimality criteria in a dataframe
        hstry_designs = np.array([]).reshape((0, self.features + 1))  # all final designs in a dataframe

        for epoch in range(self.epochs):
            design_matrix = self.gen_random_design()
            for exp in range(self.experiments):
                for feat in range(self.features):
                    coordinate_opt_cr = []
                    for count, level in enumerate(self.levels[feat]):

                        # check all possible levels for the specific experiment, feature
                        design_matrix.iat[exp, feat] = level
                        model_matrix = self.gen_model_matrix(data=design_matrix)

                        criterion = self.d_opt(model_matrix)
                        coordinate_opt_cr.append(criterion)

                    hstry_opt_cr.append([epoch, exp, feat, *coordinate_opt_cr])
                    # updated design_matrix
                    design_matrix.iat[exp, feat] = self.levels[feat][coordinate_opt_cr.index(max(coordinate_opt_cr))]

            # clean results of inner loops
            hstry_designs = np.append(hstry_designs,
                                      np.hstack((np.array([epoch]*self.experiments).reshape(-1, 1),
                                                 np.array(design_matrix))),
                                      axis=0)

        hstry_designs, hstry_opt_cr = self.clear_histories(optimalities=hstry_opt_cr, designs=hstry_designs, design_mat=design_matrix)
        best_design = self.find_best_design(histories=hstry_opt_cr, designs=hstry_designs)
        model_matrix = self.gen_model_matrix(data=best_design)

        return best_design, model_matrix,  hstry_designs, hstry_opt_cr
