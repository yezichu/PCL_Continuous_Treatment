import random
import os
import numpy as np
import torch


def set_random_seed(seed):
    """
    Set the seed for random number generators in Python, NumPy, and PyTorch.

    Args:
        seed (int): The seed value for ensuring reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_array_to_npy(array, filename_prefix, cfg):
    """
    Save a NumPy array to a .npy file.

    Args:
        array (ndarray): The NumPy array to save.
        filename_prefix (str): The prefix for the filename.
        cfg (Namespace): Configuration object containing various attributes like model, dataset, sample, and ATE directory.
    """
    save_filename = f"{cfg.model}_{cfg.dataset}_{cfg.sample}_{filename_prefix}"
    save_path = os.path.join(cfg.ATE, save_filename)
    if not os.path.exists(cfg.ATE):
        os.makedirs(cfg.ATE)
    np.save(save_path, array)


def causal_mse(pre, tar) -> float:
    """
    Compute the mean squared error (MSE) between the predicted and target values.

    Args:
        pre (np.ndarray): The predicted values, typically of shape (seeds, n_outputs).
        tar (np.ndarray): The target or true values, typically of shape (n_outputs,).

    Returns:
        float: The row-wise mean squared error, typically of shape (seeds,) .
    """
    difference = pre - tar
    row_average = np.mean(difference**2, axis=1)
    return row_average


def compute_mean_and_variance(results):
    """
    Calculate the mean and variance of an array of results.

    Args:
        results (np.ndarray): An array of shape (n,) containing the results for n seeds.

    Returns:
        tuple: A tuple containing the mean and variance of the results.
    """
    mean_result = np.mean(results)
    variance_result = np.var(results)

    return mean_result, variance_result
