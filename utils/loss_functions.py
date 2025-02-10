"""
Code adapted from: https://github.com/kratzert/ealstm_regional_modeling/blob/master/papercode/nseloss.py
"""

import torch


class NSELoss(torch.nn.Module):
    """
    Calculates the batch-wise Nash-Sutcliffe Efficiency (NSE) Loss.

    Each sample is weighted by 1 / (std + eps)^2, where std is the standard deviation
    of the discharge from the corresponding basin.

    Attributes:
        eps (float): A small constant added for numerical stability and smoothing.

    Args:
        eps (float, optional): Constant added for numerical stability. Defaults to 0.1.
    """

    def __init__(self, eps: float = 0.1):
        super(NSELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Computes the batch-wise NSE Loss.

        Args:
            y_pred (torch.Tensor): Predicted discharge values.
            y_true (torch.Tensor): True discharge values.
            q_stds (torch.Tensor): Standard deviation of discharge (calculated over the training period) for each sample.

        Returns:
            torch.Tensor: The computed NSE Loss.
        """
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()

        # Calculate mean of true values
        y_true_mean = torch.mean(y_true, dim=0)

        # Calculate numerator and denominator
        numerator = torch.sum(torch.square(y_pred - y_true))
        denominator = torch.sum(torch.square(y_true - y_true_mean))

        # Calculate NSE loss
        nse = numerator / (denominator + self.eps)

        return nse
