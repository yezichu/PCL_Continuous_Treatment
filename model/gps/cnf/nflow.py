import torch.nn as nn


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow model interface.
    """

    def __init__(self, layers, prior):
        """
        Initializes the Normalizing Flow model with the specified layers and prior distribution.

        Args:
            layers (list): List of `InvertibleLayer` objects.
            prior (torch.distributions.Distribution): The prior distribution for the latent variable.
        """
        super(NormalizingFlow, self).__init__()

        self.layers = nn.ModuleList(layers)
        self.prior = prior

    def log_prob(self, X, C):
        """
        alculates the loss function.

        Args:
            X (Tensor): torch.Tensor of shape [batch_size, var_size] Input sample to transform.
            C (Tensor): torch.Tensor of shape [batch_size, cond_size] or None Condition values.

        Returns:
            log_likelihood (Tensor): Calculated log likelihood.
        """
        log_likelihood = None

        for layer in self.layers:
            X, change = layer.f(X, C)
            if log_likelihood is not None:
                log_likelihood = log_likelihood + change
            else:
                log_likelihood = change
        log_likelihood = log_likelihood + self.prior.log_prob(X)

        return log_likelihood

    def sample(self, C):
        """
        Sample new objects based on the give conditions.

        Args:
            C (Tensor): torch.Tensor of shape [batch_size, cond_size] or Int Condition values or number of samples to generate.

        Returns:
            X (Tensor): torch.Tensor of shape [batch_size, var_size] Generated sample.
        """
        if type(C) == type(1):
            n = C
            C = None
        else:
            n = len(C)

        X = self.prior.sample((n,))
        for layer in self.layers[::-1]:
            X = layer.g(X, C)

        return X
