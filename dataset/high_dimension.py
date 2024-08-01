import torch
import numpy as np
from scipy.sparse import diags
from utils.data_class import PVTrainDataSet, PVTestDataSet


def Lambda(t):
    return (0.9-0.1)*np.exp(t)/(1+np.exp(t)) + 0.1


class High_dim:
    def __init__(self, seeds=43, size=1000, dim_z=1, dim_w=3, dim_x=10, type='quardratic'):
        """
        Initialize the HighDim class with the specified parameters.

        Args:
            seeds (int): Random seed for reproducibility.
            size (int): Number of samples to generate.
            dim_z (int): Dimension of Z.
            dim_w (int): Dimension of W.
            dim_x (int): Dimension of X.
            model_type (str): Type of the model ('quadratic', 'peaked', 'sigmoid').
        """
        self.seeds = seeds
        self.size = size
        self.dim_z = dim_z
        self.dim_w = dim_w
        self.dim_x = dim_x
        self.type = type

    def generatate_high(self, totensor: bool = False,):
        """
        Generate synthetic data with the specified dimensions and model type.

        Args:
            totensor (bool): If True, convert data to PyTorch tensors.
            test (bool): If True, generate test data.

        Returns:
            PVTrainDataSet or PVTestDataSet: Generated data.
        """
        np.random.seed(self.seeds)

        # Generate noise terms
        e1 = np.random.normal(0, 1, self.size)
        e2 = np.random.normal(0, 1, self.size)
        e3 = np.random.normal(0, 1, self.size)
        vz = np.random.uniform(-1, 1, size=(self.size, self.dim_z))
        vw = np.random.uniform(-1, 1, size=(self.size, self.dim_w))

        Uz = e1 + e3
        Uw = e2 + e3

        # Generate Z and W variables
        Z = vz + 0.25 * np.repeat(Uz.reshape(-1, 1), self.dim_z, axis=1)
        W = vw + 0.25 * np.repeat(Uw.reshape(-1, 1), self.dim_w, axis=1)

        # Generate X with a specific covariance structure
        rho = 0.5
        k = [rho*np.ones(self.dim_x-1), np.ones(self.dim_x),
             rho*np.ones(self.dim_x-1)]
        offset = [-1, 0, 1]
        sigma = diags(k, offset).toarray()
        X = np.random.multivariate_normal(
            np.zeros(self.dim_x), sigma, size=[self.size,])

        # Define theta vectors
        theta_x = np.array([(1/(l**2))
                           for l in list(range(1, (self.dim_x+1)))])
        theta_w = np.array([(1/(l**2))
                           for l in list(range(1, (self.dim_w+1)))])
        theta_z = np.array([(1/(l**2))
                           for l in list(range(1, (self.dim_z+1)))])

        # Calculate treatment variable A
        A = Lambda(3*X@theta_x + 3*Z@theta_z) + 0.25*Uw

        # Calculate outcome variable Y based on the model type
        if self.type == 'quardratic':
            structure = 1.2*A + (A**2)
        elif self.type == 'peaked':
            structure = 2*(A**4/600 + np.exp(-4*A**2) + A/10 - 2) + 1.2*A
        elif self.type == 'sigmoid':
            structure = np.log(abs(16*A-8)+1)*np.sign(A-0.5) + 1.2*A
        Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:, 0] + 0.25*Uz

        # Convert to PyTorch tensors if requested
        if totensor:
            return PVTrainDataSet(treatment=torch.tensor(A[:, np.newaxis], dtype=torch.float32),
                                  treatment_proxy=torch.tensor(
                                      Z, dtype=torch.float32),
                                  outcome_proxy=torch.tensor(
                                      W, dtype=torch.float32),
                                  outcome=torch.tensor(
                                      Y[:, np.newaxis], dtype=torch.float32),
                                  backdoor=torch.tensor(X, dtype=torch.float32))
        else:
            return PVTrainDataSet(treatment=A[:, np.newaxis],
                                  treatment_proxy=Z,
                                  outcome_proxy=W,
                                  outcome=Y[:, np.newaxis],
                                  backdoor=X)

    def generate_test(self, size, seed=43, totensor=False) -> None:
        """
        Generate test data with specified parameters.

        Args:
            size (int): Number of test samples to generate.
            seed (int): Random seed for reproducibility.
            totensor (bool): If True, convert data to PyTorch tensors.

        Returns:
            PVTestDataSet: Generated test data.
        """
        np.random.seed(seed)
        e1 = np.random.normal(0, 1, size)
        e2 = np.random.normal(0, 1, size)
        e3 = np.random.normal(0, 1, size)
        vz = np.random.uniform(-1, 1, size=(size, self.dim_z))
        vw = np.random.uniform(-1, 1, size=(size, self.dim_w))

        Uz = e1 + e3
        Uw = e2 + e3

        Z = vz + 0.25 * np.repeat(Uz.reshape(-1, 1), self.dim_z, axis=1)
        W = vw + 0.25 * np.repeat(Uw.reshape(-1, 1), self.dim_w, axis=1)

        rho = 0.5
        k = [rho*np.ones(self.dim_x-1), np.ones(self.dim_x),
             rho*np.ones(self.dim_x-1)]
        offset = [-1, 0, 1]
        sigma = diags(k, offset).toarray()
        X = np.random.multivariate_normal(
            np.zeros(self.dim_x), sigma, size=[size,])

        theta_x = np.array([(1/(l**2))
                           for l in list(range(1, (self.dim_x+1)))])
        theta_w = np.array([(1/(l**2))
                           for l in list(range(1, (self.dim_w+1)))])
        theta_z = np.array([(1/(l**2))
                           for l in list(range(1, (self.dim_z+1)))])

        A = Lambda(3*X@theta_x + 3*Z@theta_z) + 0.25*Uw

        if self.type == 'quardratic':
            structure = 1.2*A + (A**2)

            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:, 0] + 0.25*Uz
        elif self.type == 'peaked':
            structure = 2*(A**4/600 + np.exp(-4*A**2) + A/10 - 2) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:, 0] + 0.25*Uz
        elif self.type == 'sigmoid':
            structure = np.log(abs(16*A-8)+1)*np.sign(A-0.5) + 1.2*A
            Y = structure + 1.2*(X@theta_x + W@theta_w) + A*X[:, 0] + 0.25*Uz

        if totensor:
            return PVTestDataSet(treatment=torch.tensor(A[:, np.newaxis], dtype=torch.float32),
                                 treatment_proxy=torch.tensor(
                                     Z, dtype=torch.float32),
                                 outcome_proxy=torch.tensor(
                                     W, dtype=torch.float32),
                                 outcome=torch.tensor(
                                     Y[:, np.newaxis], dtype=torch.float32),
                                 backdoor=torch.tensor(X, dtype=torch.float32))
        else:
            return PVTestDataSet(treatment=A[:, np.newaxis],
                                 treatment_proxy=Z,
                                 outcome_proxy=W,
                                 outcome=Y[:, np.newaxis],
                                 backdoor=X)

    @staticmethod
    def generate_test_effect(a, b, c, type, dim_z, dim_w, dim_x):
        """
        Generate test effects for a range of treatment values A.

        Args:
            a (float): Lower bound of A.
            b (float): Upper bound of A.
            c (int): Number of samples in A.
            model_type (str): Type of the model ('quadratic', 'peaked', 'sigmoid').
            dim_z (int): Dimension of Z.
            dim_w (int): Dimension of W.
            dim_x (int): Dimension of X.

        Returns:
            tuple: Arrays of treatment values A and their corresponding effects.
        """
        A = np.linspace(a, b, c)
        e1 = np.random.normal(0, 1, 10000)
        e2 = np.random.normal(0, 1, 10000)
        e3 = np.random.normal(0, 1, 10000)
        vz = np.random.uniform(-1, 1, size=(10000, dim_z))
        vw = np.random.uniform(-1, 1, size=(10000, dim_w))

        Uz = e1 + e3
        Uw = e2 + e3

        Z = vz + 0.25 * np.repeat(Uz.reshape(-1, 1), dim_z, axis=1)
        W = vw + 0.25 * np.repeat(Uw.reshape(-1, 1), dim_w, axis=1)

        rho = 0.5,
        k = [rho*np.ones(dim_x-1), np.ones(dim_x), rho*np.ones(dim_x-1)]
        offset = [-1, 0, 1]
        sigma = diags(k, offset).toarray()
        X = np.random.multivariate_normal(
            np.zeros(dim_x), sigma, size=[10000,])

        theta_x = np.array([(1/(l**2)) for l in list(range(1, (dim_x+1)))])
        theta_w = np.array([(1/(l**2)) for l in list(range(1, (dim_w+1)))])
        theta_z = np.array([(1/(l**2)) for l in list(range(1, (dim_z+1)))])

        if type == 'quardratic':
            treatment_effect = np.array([np.mean(
                1.2*a + (a**2) + 1.2*(X@theta_x + W@theta_w) + a*X[:, 0] + 0.25*Uz) for a in A])
        elif type == 'peaked':
            treatment_effect = np.array([np.mean(2*(a**4/600 + np.exp(-4*a**2) + a/10 - 2) + 1.2*a + 1.2*(
                X@theta_x + W@theta_w) + a*X[:, 0] + 0.25*Uz) for a in A])
        elif type == 'sigmoid':
            treatment_effect = np.array([np.mean(np.log(abs(16*a-8)+1)*np.sign(
                a-0.5) + 1.2*a + 1.2*(X@theta_x + W@theta_w) + a*X[:, 0] + 0.25*Uz) for a in A])

        return A, treatment_effect
