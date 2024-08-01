import torch
import torch.nn as nn
from model.gps.cnf.nflow import NormalizingFlow
from torch.utils.data import TensorDataset, DataLoader


def gen_network(n_inputs, n_outputs, hidden=(10,), activation='tanh'):
    """
    Generates a neural network with specified input, output, hidden layers, and activation function.

    Args:
        n_inputs (int): Number of input features.
        n_outputs (int): Number of output features.
        hidden (tuple, optional): Number of neurons in hidden layers. Default is (10,).
        activation (str, optional): Activation function. Possible values: 'tanh', 'relu', 'leakyrelu'. Default is 'tanh'.

    Returns:
        model (nn.Sequential): The constructed neural network.
    """
    model = nn.Sequential()
    for i in range(len(hidden)):

        # add layer
        if i == 0:
            alayer = nn.Linear(n_inputs, hidden[i])
        else:
            alayer = nn.Linear(hidden[i-1], hidden[i])
        model.append(alayer)
        model.append(nn.Dropout(0.2))

        # add activation
        if activation == 'tanh':
            act = nn.Tanh()
        elif activation == 'relu':
            act = nn.ReLU()
        elif activation == 'leakyrelu':
            act = nn.LeakyReLU()
        else:
            act = nn.ReLU()
        model.append(act)

    # output layer
    model.append(nn.Linear(hidden[-1], n_outputs))

    return model


class CNFLayer(nn.Module):
    """
    Invertible RealNVP function for RealNVP normalizing flow model.
    """

    def __init__(self, DEVICE, var_size, cond_size, mask, hidden=(10,), activation='tanh'):
        """
        Initializes the Normalizing Flow model.

        Args:
            DEVICE (str): Device to run the model ('cpu' or 'cuda').
            var_size (int): Input vector size.
            cond_size (int): Conditional vector size.
            mask (Tensor): Tensor of {0, 1} to separate input vector components into two groups. Example: [0, 1, 0, 1].
            hidden (tuple, optional): Number of neurons in hidden layers. Example: (10, 20, 15).
            activation (str, optional): Activation function of the hidden neurons. Possible values: 'tanh', 'relu'.
        """
        super(CNFLayer, self).__init__()

        self.mask = mask.to(DEVICE)
        self.nn_t = gen_network(var_size + cond_size,
                                var_size, hidden, activation)
        self.nn_s = gen_network(var_size + cond_size,
                                var_size, hidden, activation)

    def f(self, X, C=None):
        """
        Implementation of forward pass.

        Args:
            X (Tensor): torch.Tensor of shape [batch_size, var_size] Input sample to transform.
            C (Tensor, optional): torch.Tensor of shape [batch_size, cond_size] or None Condition values.


        Returns:
            new_X (Tensor): torch.Tensor of shape [batch_size, var_size] Transformed X.
            log_det (Tensor): torch.Tensor of shape [batch_size] Logarithm of the Jacobian determinant.
        """
        if C is not None:
            XC = torch.cat((X * self.mask[None, :], C), dim=1)
        else:
            XC = X * self.mask[None, :]

        T = self.nn_t(XC)
        S = self.nn_s(XC)

        X_new = (X * torch.exp(S) + T) * \
            (1 - self.mask[None, :]) + X * self.mask[None, :]
        log_det = (S * (1 - self.mask[None, :])).sum(dim=-1)
        return X_new, log_det

    def g(self, X, C=None):
        """
        Implementation of backward (inverse) pass.

        Args:
            X (Tensor): torch.Tensor of shape [batch_size, var_size] Input sample to transform.
            C (Tensor, optional): torch.Tensor of shape [batch_size, cond_size] or None Condition values.

        Returns:
            new_X (Tensor): torch.Tensor of shape [batch_size, var_size] Transformed X.
        """
        if C is not None:
            XC = torch.cat((X * self.mask[None, :], C), dim=1)
        else:
            XC = X * self.mask[None, :]

        T = self.nn_t(XC)
        S = self.nn_s(XC)

        X_new = ((X - T) * torch.exp(-S)) * \
            (1 - self.mask[None, :]) + X * self.mask[None, :]
        return X_new


class CNF:
    """
    RealNVP-based normalizing flow model.
    """

    def __init__(self, DEVICE, n_layers=8, hidden=(10,), activation='tanh',
                 batch_size=32, n_epochs=10, lr=0.0001, weight_decay=0):
        """
        Initializes the RealNVP normalizing flow model.

        Args:
            DEVICE (str): Device to run the model ('cpu' or 'cuda').
            n_layers (int): Number of RealNVP layers.
            hidden (tuple, optional): Number of neurons in hidden layers. Example: (10,).
            activation (str, optional): Activation function of the hidden neurons. Possible values: 'tanh', 'relu'.
            batch_size (int, optional): Batch size. Default is 32.
            n_epochs (int, optional): Number of epoches for fitting the model. Default is 10.
            lr (float, optional): Learning rate. Default is 0.0001.
            weight_decay (float, optional): L2 regularization coefficient. Default is 0.
        """

        self.n_layers = n_layers
        self.hidden = hidden
        self.activation = activation
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.DEVICE = DEVICE

        self.prior = None
        self.nf = None
        self.opt = None

        self.loss_history = []
        self.val_loss = []

    def _model_init(self, X, C):
        """
        Trains the model on the given data.

        Args:
            X (Tensor): Input sample tensor.
            C (Tensor): Condition tensor.
        """

        var_size = X.shape[1]
        if C is not None:
            cond_size = C.shape[1]
        else:
            cond_size = 0

        # init prior
        if self.prior is None:
            self.prior = torch.distributions.MultivariateNormal(torch.zeros(var_size, device=self.DEVICE),
                                                                torch.eye(var_size, device=self.DEVICE))
        # init NF model and optimizer
        if self.nf is None:

            layers = []
            for i in range(self.n_layers):
                alayer = CNFLayer(DEVICE=self.DEVICE, var_size=var_size,
                                  cond_size=cond_size,
                                  mask=((torch.arange(var_size) + i) % 2),
                                  hidden=self.hidden,
                                  activation=self.activation)
                layers.append(alayer)

            self.nf = NormalizingFlow(
                layers=layers, prior=self.prior).to(self.DEVICE)
            self.opt = torch.optim.Adam(self.nf.parameters(),
                                        lr=self.lr,
                                        weight_decay=self.weight_decay)

    def fit(self, X, C=None):
        """
        Fit the model.

        Args:
            X (ndarray): Input sample to transform.
            C (ndarray, optional): Condition values.
        """

        # model init
        self._model_init(X, C)

        # numpy to tensor, tensor to dataset
        if C is not None:
            dataset = TensorDataset(X, C)
        else:
            dataset = TensorDataset(X)

        for epoch in range(self.n_epochs):
            for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=True):
                self.nf.train()
                X_batch = batch[0].to(self.DEVICE)

                X_batch += torch.randn_like(X_batch) * 0.1

                if C is not None:
                    C_batch = batch[1].to(self.DEVICE)
                else:
                    C_batch = None

                # caiculate loss
                loss = -self.nf.log_prob(X_batch, C_batch).mean()

                # optimization step
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # caiculate and store loss
                self.loss_history.append(loss.detach().cpu())

    def pob(self, X, C=None):
        """
        Sample new objects based on the give conditions.

        Args:
            X (ndarray): Condition values or number of samples to generate.
            C (ndarray, optional): Condition values or number of samples to generate.

        Returns:
            X (ndarray): Generated sample.
        """
        X, C = X.to(self.DEVICE), C.to(self.DEVICE)
        self.nf.eval()
        log_pob = self.nf.log_prob(X, C)
        pob = torch.exp(log_pob).cpu().detach()

        return pob

    def sample(self, C=100):
        """
        Sample new objects based on the give conditions.

        Args:
            C (int, optional): Condition values or number of samples to generate.

        Returns:
            X (ndarray): Generated sample.
        """
        if type(C) != type(1):
            C = torch.tensor(C, dtype=torch.float32, device=self.DEVICE)
        X = self.nf.sample(C).cpu().detach().numpy()
        return X
