import yaml
import numpy as np
from model.gps.GPS import GPS
from utils.kernel import epanechnikov_kernel
from model.rkhs.rkhs_approx import ApproxRKHSIVCV


class RKHS_Trainer():
    def __init__(self, dataset, gamma_gm, n_gamma_hqs, n_alphas, alpha_scales, cv, n_components):
        """
        Initializes the RKHS_Trainer with the specified parameters.

        Args:

            gamma_gm (float or str): Gamma parameter for the kernel of f.
            n_gamma_hqs (int): Number of gamma_hqs to try.   
            n_alphas (int): Number of alpha scales to try.
            alpha_scales (float or list): List of regularization scales.
            cv (int): Number of cross-validation folds.
            n_components (int): Number of approximation components.
        """
        self.dataset = dataset
        self.gamma_gm = gamma_gm
        self.n_gamma_hqs = n_gamma_hqs
        self.n_alphas = n_alphas
        self.alpha_scales = alpha_scales
        self.cv = cv
        self.n_components = n_components

    def _prepare_data(self):
        """
        Prepares the data by concatenating treatment, proxies, and backdoor variables.

        Returns:
            AZX: Combined array of treatment and treatment proxy with optional backdoor variables.
            AWX: Combined array of treatment and outcome proxy with optional backdoor variables.
            Y: Outcome variable.
        """
        A, Z, W, Y = self.dataset.treatment, self.dataset.treatment_proxy, self.dataset.outcome_proxy, self.dataset.outcome
        X = self.dataset.backdoor
        AZX = np.concatenate([A, Z], axis=1)
        AWX = np.concatenate([A, W], axis=1)
        if X is not None:
            AZX = np.concatenate([AZX, X], axis=1)
            AWX = np.concatenate([AWX, X], axis=1)
        return AZX, AWX, Y

    def fit_h_cv(self):
        """
        Fits the RKHS model to estimate the function h using cross-validation.
        """
        AZX, AWX, Y = self._prepare_data()
        RKHS = ApproxRKHSIVCV(gamma_gm=self.gamma_gm, n_gamma_hqs=self.n_gamma_hqs, n_alphas=self.n_alphas,
                              alpha_scales=self.alpha_scales, cv=self.cv, n_components=self.n_components)
        RKHS_h = RKHS.fit(AWX, Y, AZX, 'estimate_h')
        self.gamma_gm = RKHS_h.gamma_gm
        self.gamma_hq = RKHS_h.gamma_hq
        self.best_alpha_scale = RKHS_h.best_alpha_scale
        self.h = RKHS_h.predict
        return self

    def fit_q_cv(self, type='kde', cnf=None):
        """
        Fits the RKHS model to estimate the function q using cross-validation.

        Args:
            type (str, optional): Type of GPS method to use for estimating q. 
                              Can be 'kde' for kernel density estimation or 'cnf' for conditional normal form.
                              Defaults to 'kde'.
            cnf (str, optional): Path to a configuration file with hyperparameters for the 'cnf' method.
                             Defaults to None.

        Returns:
            self: The RKHS_Trainer instance with the fitted model.
        """
        AZX, AWX, _ = self._prepare_data()

        density = GPS(self.dataset.treatment,
                      self.dataset.treatment_proxy, self.dataset.backdoor)
        if type == 'kde':
            gps = density.kde_gps()
        elif type == 'cnf':
            with open(cnf, 'r') as file:
                hyperparams = yaml.safe_load(file)
            gps = density.cnf_gps(**hyperparams)

        RKHS = ApproxRKHSIVCV(gamma_gm=self.gamma_gm, n_gamma_hqs=self.n_gamma_hqs, n_alphas=self.n_alphas,
                              alpha_scales=self.alpha_scales, cv=self.cv, n_components=self.n_components)
        RKHS_q = RKHS.fit(AWX, gps, AZX, 'estimate_q')
        self.gamma_gm = RKHS_q.gamma_gm
        self.gamma_hq = RKHS_q.gamma_hq
        self.best_alpha_scale = RKHS_q.best_alpha_scale
        self.q = RKHS_q.predict
        return self

    def _prepare_input(self, point, W, X=None):
        """
        Prepares the input data for prediction.

        Args:
            point (float): The value of the treatment point.
            W (array-like): The outcome proxy data.
            X (array-like, optional): Additional backdoor data. Defaults to None.

        Returns:
            np.ndarray: The prepared input data concatenated from point, W, and optionally X.
        """
        point_full = np.full((len(W), 1), point)
        inp = np.concatenate([point_full, W], axis=1)
        if X is not None:
            inp = np.concatenate([inp, X], axis=1)
        return inp

    def _calculate_ate(self, pointA, sampleTest, method='h'):
        """
        Calculates the Average Treatment Effect (ATE) for given points using specified method.

        Args:
            pointA (array-like): The points at which to estimate the ATE.
            sampleTest: A sample object containing treatment, treatment proxy, outcome proxy, and outcome data.
            method (str, optional): The method to use for ATE calculation. Options are 'h', 'q', or 'dr'.
                                    Defaults to 'h'.

        Returns:
            list: The estimated ATE values for each point in pointA.
        """
        A, Z, W, Y = sampleTest.treatment, sampleTest.treatment_proxy, sampleTest.outcome_proxy, sampleTest.outcome
        X = sampleTest.backdoor
        A = A.ravel() if A.ndim == 2 else A
        Y = Y.ravel() if Y.ndim == 2 else Y

        ATE_list = []
        bandwidth = 1.5 * np.std(A) * (len(A) ** -
                                       0.2) if method in ['q', 'dr'] else None

        for a in pointA:
            if method == 'h':
                inp_h = self._prepare_input(a, W, X)
                ATE = np.mean(self.h(inp_h))

            elif method == 'q':
                inp_q = self._prepare_input(a, Z, X)
                q_azx = self.q(inp_q)
                ATE = np.mean(epanechnikov_kernel(
                    A - a, bandwidth) * q_azx * Y)

            elif method == 'dr':
                inp_h = self._prepare_input(a, W, X)
                inp_q = self._prepare_input(a, Z, X)
                q_azx = self.q(inp_q)
                ATE = np.mean((Y - self.h(inp_h)) * q_azx *
                              epanechnikov_kernel(A - a, bandwidth) + self.h(inp_h))

            ATE_list.append(ATE)

        return ATE_list

    def _htest(self, pointA, sampleTest) -> float:
        """
        Calculates the ATE using the 'h' method.

        Args:
            pointA (array-like): The points at which to estimate the ATE.
            sampleTest: A sample object containing the relevant data.

        Returns:
            float: The estimated ATE for the given points using the 'h' method.
        """
        return self._calculate_ate(pointA, sampleTest, method='h')

    def _qtest(self, pointA, sampleTest) -> float:
        """
        Calculates the ATE using the 'q' method.

        Args:
            pointA (array-like): The points at which to estimate the ATE.
            sampleTest: A sample object containing the relevant data.

        Returns:
            float: The estimated ATE for the given points using the 'q' method.
        """
        return self._calculate_ate(pointA, sampleTest, method='q')

    def _drtest(self, pointA, sampleTest) -> float:
        """
        Calculates the ATE using the 'dr' method.

        Args:
            pointA (array-like): The points at which to estimate the ATE.
            sampleTest: A sample object containing the relevant data.

        Returns:
            float: The estimated ATE for the given points using the 'dr' method.
        """
        return self._calculate_ate(pointA, sampleTest, method='dr')
