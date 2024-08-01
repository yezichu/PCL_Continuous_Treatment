import numpy as np
from scipy.stats import norm


def epanechnikov_kernel(x, h):
    """
    Compute the Epanechnikov kernel function.

    Args:
        x (float or ndarray): Input values, can be a scalar or an array.
        h (float): Bandwidth parameter, determines the width of the kernel.

    Returns:
        ndarray: Kernel function values, matching the shape of x.
    """
    k = (1/h)*(3/4)*(1-((x/h)**2))
    k = k*(np.abs(x/h) <= 1)
    return k


def gaussian_kernel(x, h):
    """
    Compute the Gaussian kernel function.

    Args:
        x (float or ndarray): Input values, can be a scalar or an array.
        h (float): Bandwidth parameter, determines the width of the kernel.

    Returns:
        ndarray: Kernel function values, matching the shape of x.
    """
    k = (1/h)*norm.pdf((x/h), 0, 1)
    return k


def binaryKernel(A, a):
    """
    Compute a binary kernel function.

    Args:
        A (ndarray): Binary matrix.
        a (ndarray): Binary vector.

    Returns:
        ndarray: Result of the binary kernel operation.
    """
    res = A.dot(a)
    res += (1 - A).dot(1 - a)
    return res
