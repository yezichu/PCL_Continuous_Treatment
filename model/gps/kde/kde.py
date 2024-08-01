import torch
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


def numpy_conversion(func):
    """
    Decorator to convert torch tensors to numpy arrays before calling the function.

    Args:
        func (callable): The function to be decorated.

    Returns:
        callable: The wrapper function that converts tensors to numpy arrays.
    """
    def wrapper(*args, **kwargs):
        processed_args = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                arg = arg.cpu().detach().numpy()
            processed_args.append(arg)
        return func(*processed_args, **kwargs)
    return wrapper


@numpy_conversion
def kde_gps(A, W, X):
    """
    Estimates the Generalized Propensity Score (GPS) using Kernel Density Estimation (KDE).

    Args:
        A (np.ndarray): Treatment variable.
        W (np.ndarray): Outcome proxy variable.
        X (np.ndarray, optional): Additional backdoor variables. Defaults to None.

    Returns:
        np.ndarray: The estimated Generalized Propensity Score (GPS).
    """
    WX = W
    AWX = np.concatenate([A, W], axis=1)
    if X is not None:
        WX = np.concatenate([WX, X], axis=1)
        AWX = np.concatenate([AWX, X], axis=1)

    bandwidths = {'bandwidth': np.logspace(-1, 0, 20)}

    grid_wx = GridSearchCV(KernelDensity(), bandwidths)
    grid_wx.fit(WX)
    bandwidth_est_wx = grid_wx.best_estimator_.bandwidth

    grid_awx = GridSearchCV(KernelDensity(), bandwidths)
    grid_awx.fit(AWX)
    bandwidth_est_awx = grid_awx.best_estimator_.bandwidth

    kde_wx = KernelDensity(kernel='gaussian', bandwidth=bandwidth_est_wx)
    kde_wx.fit(WX)
    f_wx = np.exp(kde_wx.score_samples(WX))

    kde_awx = KernelDensity(kernel='gaussian', bandwidth=bandwidth_est_awx)
    kde_awx.fit(AWX)
    f_awx = np.exp(kde_awx.score_samples(AWX))

    gps = f_wx/f_awx

    return gps


@numpy_conversion
def kde_f_a(A):
    """
    Estimates the density function of the treatment variable A using Kernel Density Estimation (KDE).

    Args:
        A (np.ndarray): Treatment variable.

    Returns:
        np.ndarray: The estimated density function values for A.
    """
    bandwidths = {'bandwidth': np.logspace(-1, 0, 20)}
    grid_a = GridSearchCV(KernelDensity(), bandwidths)
    grid_a.fit(A)
    bandwidth_est_wx = grid_a.best_estimator_.bandwidth

    kde_a = KernelDensity(kernel='gaussian', bandwidth=bandwidth_est_wx)
    kde_a.fit(A)

    f_a = np.exp(kde_a.score_samples(A))

    return f_a
