import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels, pairwise_distances


def _check_auto(param):
    """
    Check if a parameter is set to 'auto'.

    Args:
        param: The parameter to check.

    Returns:
        bool: True if the parameter is 'auto', otherwise False.
    """
    return (isinstance(param, str) and (param == 'auto'))


class _BaseRKHSIV:
    def __init__(self, *args, **kwargs):
        """
        Base class for RKHS-IV (Reproducing Kernel Hilbert Space - Instrumental Variables).

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass

    def _get_delta(self, n: int) -> float:
        """
        Compute the critical radius delta.

        Args:
            n (int): The number of samples.

        Returns:
            float: The computed delta value -> Critical radius.
        """
        delta_scale = 5 if _check_auto(self.delta_scale) else self.delta_scale
        delta_exp = .4 if _check_auto(self.delta_exp) else self.delta_exp
        return delta_scale / (n**(delta_exp))

    def _get_alpha_scale(self):
        """
        Get the alpha scale parameter.

        Returns:
            float: The alpha scale parameter.
        """
        return 60 if _check_auto(self.alpha_scale) else self.alpha_scale

    def _get_alpha_scales(self):
        """
        Get a list of alpha scales.

        Returns:
            list: A list of alpha scales.
        """
        return ([c for c in np.geomspace(0.1, 1e5, self.n_alphas)]
                if _check_auto(self.alpha_scales) else self.alpha_scales)

    def _get_alpha(self, delta, alpha_scale):
        """
        Compute the alpha parameter.

        Args:
            delta (float): The delta value.
            alpha_scale (float): The alpha scale.

        Returns:
            float: The computed alpha value.
        """
        return alpha_scale * (delta**4)

    def _get_kernel(self, X, Y=None):
        """
        Compute the kernel matrix using the specified kernel function.

        Args:
            X (array-like): The data matrix.
            Y (array-like, optional): The second data matrix.

        Returns:
            array: The computed kernel matrix.
        """
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def _get_gamma_gm(self, condition):
        """
        Compute the gamma parameter for Gaussian kernel (GM) based on median heuristic.

        Args:
            condition (array-like): The conditioning variables.

        Returns:
            float: The computed gamma parameter.
        """
        if _check_auto(self.gamma_gm):
            params = {"squared": True}
            K_condition_euclidean = pairwise_distances(
                X=condition, metric='euclidean', n_jobs=-1, **params)
            gamma_gm = 1. / \
                (np.median(
                    K_condition_euclidean[np.tril_indices(condition.shape[0], -1)]))
            return gamma_gm
        else:
            return self.gamma_gm

    def _get_kernel_gm(self, X, Y=None, gamma_gm=0.1):
        """
        Compute the Gaussian kernel matrix with specified gamma for Gaussian mechanism.

        Args:
            X (array-like): The data matrix.
            Y (array-like, optional): The second data matrix.
            gamma_gm (float, optional): The gamma parameter for Gaussian kernel.

        Returns:
            array: The computed kernel matrix.
        """
        params = {"gamma": gamma_gm}
        return pairwise_kernels(X, Y, metric='rbf', filter_params=True, **params)

    def _get_kernel_hq(self, X, Y=None, gamma_h=0.01):
        """
        Compute the kernel matrix for high-quality kernel with specified gamma.

        Args:
            X (array-like): The data matrix.
            Y (array-like, optional): The second data matrix.
            gamma_h (float, optional): The gamma parameter for high-quality kernel.

        Returns:
            array: The computed kernel matrix.
        """
        params = {"gamma": gamma_h}
        return pairwise_kernels(X, Y, metric='rbf', filter_params=True, **params)
