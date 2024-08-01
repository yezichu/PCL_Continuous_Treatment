import torch
import numpy as np
from model.gps.kde.kde import kde_gps
from model.gps.cnf.normflow import CNF

class GPS():
    def __init__(self,A,W,X=None):
        """
        Initializes the GPS class with treatment (A), outcome proxy (W), and optional backdoor variables (X).
        """
        self.A = A
        self.W = W
        self.X = X
    def kde_gps(self):
        """
        Computes the generalized propensity score using Kernel Density Estimation (KDE).
        
        Returns:
            gps: Array of generalized propensity scores.
        """
        gps = kde_gps(self.A,self.W,self.X)
        return gps
    def cnf_gps(self,device,n_layers,hidden,batch_size,lr,n_epochs,weight_decay):
        """
        Computes the generalized propensity score using a Conditional Normalizing Flow (CNF) model.
        
        Args:
            device (str): Device to use for computation (e.g., 'cpu' or 'cuda').
            n_layers (int): Number of layers in the CNF model.
            hidden (int): Number of hidden units in each layer.
            batch_size (int): Batch size for training.
            lr (float): Learning rate.
            n_epochs (int): Number of epochs for training.
            weight_decay (float): Weight decay for regularization.

        Returns:
            gps: Array of generalized propensity scores.
        """
        gps_train = CNF(DEVICE=device, n_layers=n_layers, hidden=hidden,
                        batch_size=batch_size, lr=lr, n_epochs=n_epochs, weight_decay=weight_decay)
        WX =self.W
        if self.X is not None:
            WX = np.concatenate([WX, self.X], axis=1)
        WX = torch.tensor(WX, dtype=torch.float32)
        A_torch = torch.tensor(self.A, dtype=torch.float32)
        gps_train.fit(A_torch, WX)
        gps = (1/gps_train.pob(A_torch, WX)).cpu().detach().numpy()
        return gps

