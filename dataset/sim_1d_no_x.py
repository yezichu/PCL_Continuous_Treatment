import torch
import numpy as np
from utils.data_class import PVTrainDataSet, PVTestDataSet


class Sim1d_noX:
    def __init__(self, seeds=43, size=1000):
        """
        Initializes the simulation class with a random seed and sample size.

        Args:
            seeds (int): The random seed for reproducibility.
            size (int): The number of samples to generate.
        """
        self.seeds = seeds
        self.size = size

    def generatate_sim(self, totensor: bool = False, beta=1, sigma=1, W_miss=False, Z_miss=False):
        """
        Generates synthetic simulation data.

        Args:
            totensor (bool): If True, returns data as PyTorch tensors.
            beta (float): The standard deviation of the normal distribution for A.
            sigma (float): The standard deviation of the normal distribution for noise.
            W_miss (bool): If True, apply missingness transformation to W.
            Z_miss (bool): If True, apply missingness transformation to Z.

        Returns:
            PVTrainDataSet: A dataset containing treatment, treatment proxy, outcome proxy, and outcome.
        """
        np.random.seed(self.seeds)

        # Generate latent variables
        U2 = np.random.uniform(-1, 2, size=self.size)
        U1 = np.random.uniform(0, 1, size=self.size) - \
            ((U2 >= 0) & (U2 <= 1)).astype(int)

        # Generate proxies for treatment and outcome
        Z2 = U2 + np.random.uniform(-1, 1, size=self.size)
        Z1 = U1 + np.random.normal(0, sigma, size=self.size)
        Z = np.c_[Z1, Z2]
        if Z_miss:
            Z = np.sqrt(np.abs(Z)) + 1

        W1 = U1 + np.random.uniform(-1, 1, size=self.size)
        W2 = U2 + np.random.normal(0, sigma, size=self.size)
        W = np.c_[W1, W2]
        if W_miss:
            W = np.sqrt(np.abs(W)) + 1

        # Generate treatment and outcome
        A = U2 + np.random.normal(0, beta, size=self.size)
        Y = 3*np.cos(2 * (0.3 * U1 + 0.3 * U2 + 0.2)+1.5*A) + \
            np.random.normal(0, 1, size=self.size)

        # Convert to PyTorch tensors if specified
        if totensor:
            return PVTrainDataSet(treatment=torch.tensor(A[:, np.newaxis], dtype=torch.float32),
                                  treatment_proxy=torch.tensor(
                                      Z, dtype=torch.float32),
                                  outcome_proxy=torch.tensor(
                                      W, dtype=torch.float32),
                                  outcome=torch.tensor(
                                      Y[:, np.newaxis], dtype=torch.float32),
                                  backdoor=None)
        else:
            return PVTrainDataSet(treatment=A[:, np.newaxis],
                                  treatment_proxy=Z,
                                  outcome_proxy=W,
                                  outcome=Y[:, np.newaxis],
                                  backdoor=None)

    @staticmethod
    def generate_test(size, seed=43, totensor=False, beta=1, sigma=1):
        """
        Generates synthetic test data.

        Args:
            size (int): The number of samples to generate.
            seed (int): The random seed for reproducibility.
            totensor (bool): If True, returns data as PyTorch tensors.
            beta (float): The standard deviation of the normal distribution for A.
            sigma (float): The standard deviation of the normal distribution for noise.

        Returns:
            PVTestDataSet: A dataset containing treatment, treatment proxy, outcome proxy, and outcome.
        """
        np.random.seed(seed)

        # Generate latent variables
        U2 = np.random.uniform(-1, 2, size=size)
        U1 = np.random.uniform(0, 1, size=size) - \
            ((U2 >= 0) & (U2 <= 1)).astype(int)

        # Generate proxies for treatment and outcome
        Z1 = U1 + np.random.normal(0, sigma, size=size)
        Z2 = U2 + np.random.uniform(-1, 1, size=size)
        Z = np.c_[Z1, Z2]

        W1 = U1 + np.random.uniform(-1, 1, size=size)
        W2 = U2 + np.random.normal(0, sigma, size=size)
        W = np.c_[W1, W2]

        # Generate treatment and outcome
        A = U2 + np.random.normal(0, beta, size=size)
        Y = 3 * np.cos(2 * (0.3 * U1 + 0.3 * U2 + 0.2) + 1.5 *
                       A) + np.random.normal(0, 1, size=size)

        # Convert to PyTorch tensors if specified
        if totensor:
            return PVTestDataSet(
                treatment=torch.tensor(A[:, np.newaxis], dtype=torch.float32),
                treatment_proxy=torch.tensor(Z, dtype=torch.float32),
                outcome_proxy=torch.tensor(W, dtype=torch.float32),
                outcome=torch.tensor(Y[:, np.newaxis], dtype=torch.float32),
                backdoor=None
            )
        else:
            return PVTestDataSet(
                treatment=A[:, np.newaxis],
                treatment_proxy=Z,
                outcome_proxy=W,
                outcome=Y[:, np.newaxis],
                backdoor=None
            )

    @staticmethod
    def generate_test_effect(a, b, c):
        """
        Computes the average treatment effect over a range of treatment values.

        Args:
            a (float): The start value of the treatment.
            b (float): The end value of the treatment.
            c (int): The number of points to evaluate.

        Returns:
            tuple: Arrays of treatment values and corresponding effects.
        """
        A = np.linspace(a, b, c)
        U2 = np.random.uniform(-1, 2, size=10000)
        U1 = np.random.uniform(0, 1, size=10000) - \
            ((U2 >= 0) & (U2 <= 1)).astype(int)
        treatment_effect = np.array(
            [np.mean(3 * np.cos(2 * (0.3 * U1+0.3 * U2 + 0.2) + 1.5*a)) for a in A])
        return A, treatment_effect
