import sklearn
import numpy as np
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_X_y
from sklearn.preprocessing import RobustScaler as Scaler
from sklearn.kernel_approximation import Nystroem, RBFSampler
from model.rkhs.hyper_parameter import _BaseRKHSIV, _check_auto


class ApproxRKHSIV(_BaseRKHSIV):

    def __init__(self, kernel_approx='nystrom', n_components=25,
                 gamma_gm='auto', gamma_hq=0.1,
                 delta_scale='auto', delta_exp='auto', alpha_scale='auto'):
        """
        Approximate RKHS-IV model with kernel approximation.

        Args:
            kernel_approx (str): Approximation method; 'nystrom' or 'rbfsampler'.
            n_components (int): Number of approximation components.
            gamma_gm (float or str): Gamma parameter for the kernel of f.
            gamma_hq (float): Gamma parameter for the kernel of h.
            delta_scale (float or str): Scale of the critical radius.
            delta_exp (float or str): Exponent of the critical radius.
            alpha_scale (float or str): Scale of the regularization.
        """
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma_gm = gamma_gm
        self.gamma_hq = gamma_hq
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scale = alpha_scale

    def _get_new_approx_instance(self, gamma):
        """
        Create a new instance of the kernel approximation method.

        Args:
            gamma (float): Gamma parameter for the kernel.

        Returns:
            Kernel approximator instance.
        """
        if self.kernel_approx == 'rbfsampler':
            return RBFSampler(gamma=gamma, n_components=self.n_components, random_state=1)
        elif self.kernel_approx == 'nystrom':
            return Nystroem(kernel='rbf', gamma=gamma, random_state=1, n_components=self.n_components)
        else:
            raise AttributeError("Invalid kernel approximator")

    def fit(self, AWX, model_target, AZX, type):
        """
        Fit the ApproxRKHSIV model.

        Args:
            AWX (array-like): Array of features for treatment effects.
            model_target (array-like): Target variable.
            AZX (array-like): Array of features for condition variables.
            type (str): Type of estimation; 'estimate_h' or 'estimate_q'.

        Returns:
            self: Fitted model instance.
        """
        # Determine the inputs and condition based on the type
        if type == 'estimate_h':
            X = AWX
            condition = AZX
        else:
            X = AZX
            condition = AWX
        y = model_target

        # Check and standardize data
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)
        condition = Scaler().fit_transform(condition)

        # Compute gamma_gm and RootKf
        gamma_gm = self._get_gamma_gm(condition=condition)
        self.gamma_gm = gamma_gm
        self.featCond = self._get_new_approx_instance(gamma=self.gamma_gm)
        RootKf = self.featCond.fit_transform(condition)

        # Standardize X and compute gamma_hq and RootKh
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        self.featX = self._get_new_approx_instance(gamma=self.gamma_hq)
        RootKh = self.featX.fit_transform(X)

        # Compute delta and alpha
        n = X.shape[0]
        delta = self._get_delta(n)
        alpha = self._get_alpha(delta, self._get_alpha_scale())

        # Compute Q, A, W, B for the final model parameters
        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ y
        self.a = np.linalg.lstsq(W, B, rcond=None)[0]
        self.fitted_delta = delta
        return self

    def predict(self, X):
        """
        Predict the target values for given input data.

        Args:
            X (array-like): Input data.

        Returns:
            array: Predicted values.
        """
        X = self.transX.transform(X)
        return self.featX.transform(X) @ self.a


class ApproxRKHSIVCV(ApproxRKHSIV):

    def __init__(self, kernel_approx='nystrom', n_components=25,
                 gamma_gm='auto', gamma_hqs='auto', n_gamma_hqs=10,
                 delta_scale='auto', delta_exp='auto', alpha_scales='auto', n_alphas=30, cv=6):
        """
        Approximate RKHS-IV model with cross-validation.

        Args:
            kernel_approx (str): Approximation method; 'nystrom' or 'rbfsampler'.
            n_components (int): Number of approximation components.
            gamma_gm (float or str): Gamma parameter for the kernel of f.
            gamma_hqs (float or list): List of gamma parameters for the kernel of h.
            n_gamma_hqs (int): Number of gamma_hqs to try.
            delta_scale (float or str): Scale of the critical radius.
            delta_exp (float or str): Exponent of the critical radius.
            alpha_scales (float or list): List of regularization scales.
            n_alphas (int): Number of alpha scales to try.
            cv (int): Number of cross-validation folds.
        """
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.gamma_gm = gamma_gm
        self.gamma_hqs = gamma_hqs
        self.n_gamma_hqs = n_gamma_hqs
        self.delta_scale = delta_scale
        self.delta_exp = delta_exp
        self.alpha_scales = alpha_scales
        self.n_alphas = n_alphas
        self.cv = cv

    def _get_gamma_hqs(self, X):
        """
        Compute gamma_hqs values for the given data.

        Args:
            X (array-like): Input data.

        Returns:
            list: List of gamma_hqs values.
        """
        if _check_auto(self.gamma_hqs):
            params = {"squared": True}
            K_X_euclidean = sklearn.metrics.pairwise_distances(
                X=X, metric='euclidean', **params)
            return 1./np.quantile(K_X_euclidean[np.tril_indices(X.shape[0], -1)], np.array(range(1, self.n_gamma_hqs))/self.n_gamma_hqs)
        else:
            return self.gamma_hqs

    def fit(self, AWX, model_target, AZX, type):
        """
        Fit the ApproxRKHSIVCV model with cross-validation.

        Args:
            AWX (array-like): Array of features for treatment effects.
            model_target (array-like): Target variable.
            AZX (array-like): Array of features for condition variables.
            type (str): Type of estimation; 'estimate_h' or 'estimate_q'.

        Returns:
            self: Fitted model instance.
        """
        model_target = model_target.ravel() if model_target.ndim == 2 else model_target
        X, y = (AWX, model_target) if type == 'estimate_h' else (
            AZX, model_target)
        condition = AZX if type == 'estimate_h' else AWX

        # Check and standardize data
        X, y = check_X_y(X, y, accept_sparse=True)
        condition, y = check_X_y(condition, y, accept_sparse=True)
        condition = Scaler().fit_transform(condition)

        # Compute gamma_gm and RootKf
        gamma_gm = self._get_gamma_gm(condition=condition)
        self.gamma_gm = gamma_gm
        self.featCond = self._get_new_approx_instance(gamma=gamma_gm)
        RootKf = self.featCond.fit_transform(condition)

        # Standardize X and compute gamma_hqs and RootKhs
        self.transX = Scaler()
        self.transX.fit(X)
        X = self.transX.transform(X)
        gamma_hqs = self._get_gamma_hqs(X)
        RootKhs = [self._get_new_approx_instance(
            gamma=gammah).fit_transform(X) for gammah in gamma_hqs]

        # Cross-validation to select best alpha and gamma_hq
        n = X.shape[0]
        alpha_scales = self._get_alpha_scales()
        n_train = n * (self.cv - 1) / self.cv
        n_test = n / self.cv
        delta_train = self._get_delta(n_train)
        delta_test = self._get_delta(n_test)

        scores = []
        for it1, (train, test) in enumerate(KFold(n_splits=self.cv).split(X)):
            RootKf_train, RootKf_test = RootKf[train], RootKf[test]
            Q_train = np.linalg.pinv(
                RootKf_train.T @ RootKf_train / (2 * n_train * (delta_train**2)) + np.eye(self.n_components) / 2)
            Q_test = np.linalg.pinv(
                RootKf_test.T @ RootKf_test / (2 * n_test * (delta_test**2)) + np.eye(self.n_components) / 2)
            scores.append([])
            for it2, RootKh in enumerate(RootKhs):
                RootKh_train, RootKh_test = RootKh[train], RootKh[test]
                A_train = RootKh_train.T @ RootKf_train
                AQA_train = A_train @ Q_train @ A_train.T
                B_train = A_train @ Q_train @ RootKf_train.T @ y[train]
                scores[it1].append([])
                for alpha_scale in alpha_scales:
                    alpha = self._get_alpha(delta_train, alpha_scale)
                    a = np.linalg.lstsq(AQA_train + alpha *
                                        np.eye(self.n_components), B_train, rcond=None)[0]
                    res = RootKf_test.T @ (y[test] - RootKh_test @ a)
                    scores[it1][it2].append(
                        (res.T @ Q_test @ res).reshape(-1)[0] / (len(test)**2))

        avg_scores = np.mean(np.array(scores), axis=0)
        best_ind = np.unravel_index(np.argmin(avg_scores), avg_scores.shape)

        # Set the best parameters
        self.gamma_hq = gamma_hqs[best_ind[0]]
        self.featX = self._get_new_approx_instance(gamma=self.gamma_hq)
        RootKh = self.featX.fit_transform(X)
        self.best_alpha_scale = alpha_scales[best_ind[1]]
        delta = self._get_delta(n)
        self.best_alpha = self._get_alpha(delta, self.best_alpha_scale)

        # Compute Q, A, W, B for the final model parameters
        Q = np.linalg.pinv(RootKf.T @ RootKf /
                           (2 * n * delta**2) + np.eye(self.n_components) / 2)
        A = RootKh.T @ RootKf
        W = (A @ Q @ A.T + self.best_alpha * np.eye(self.n_components))
        B = A @ Q @ RootKf.T @ y
        self.a = np.linalg.lstsq(W, B, rcond=None)[0]
        self.fitted_delta = delta
        return self
