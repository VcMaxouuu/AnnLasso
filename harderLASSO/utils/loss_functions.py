"""
Contains the different loss functions for different tasks.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple

def custom_penalty(penalized_tensor: torch.Tensor, nu: float) -> torch.Tensor:
    """
    If 'nu' is not None, computes the custom non-convex penalty given by:

    .. math:: P_\nu(\boldsymbol{\theta})=\sum_{j=1}^p \rho_\nu\left(\theta_j\right) \quad\text{with}\quad \rho_\nu(\theta)=\frac{|\theta|}{1+|\theta|^{1-\nu}}

    Else simply computes the l1 penalty.

    :param penalized_tensor: The tensor to which the penalty is applied (e.g., model weights).
    :type penalized_tensor: torch.Tensor
    :param nu: The exponent parameter.
    :type nu: float
    :return: The computed penalty term.
    :rtype: torch.Tensor
    """

    epsilon = 1e-8
    if nu is not None:
        pow_term = (torch.abs(penalized_tensor) + epsilon).pow(1 - nu)
        penalty = torch.sum(torch.abs(penalized_tensor) / (1 + pow_term))
    else:
        penalty = torch.sum(torch.abs(penalized_tensor))
    return penalty


class PenalizedLoss(nn.Module):
    """
    Base class for penalized loss functions with an added penalty term.

    :param lambda_: The regularization coefficient for the penalty term.
    :type lambda_: torch.Tensor
    :param nu: The exponent parameter for the non-convex penalty. If ``None``, the penalty reduces to L1 regularization.
    :type nu: Optional[float]
    """
    def __init__(self, lambda_: torch.Tensor, nu: Optional[float]):
        super(PenalizedLoss, self).__init__()
        self.lambda_ = lambda_
        self.nu = nu

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        penalized_tensors: List[Tuple[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the penalized loss.

        :param input: The model predictions.
        :type input: torch.Tensor
        :param target: The ground truth targets. Can be a tuple `(y, c)` for survival analysis.
        :type target: torch.Tensor
        :param penalized_tensors: A list of tuples containing the name and tensor to penalize (e.g., model parameters).
        :type penalized_tensors: List[Tuple[str, torch.Tensor]]
        :return: A tuple containing the total penalized loss (`loss + lambda_ * penalty`) and the original loss.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        loss = self.compute_loss(input, target)
        if penalized_tensors is not None:
            penalty = torch.tensor(0.0, device=input.device, requires_grad=True)
            for _, param in penalized_tensors:
                penalty = penalty + custom_penalty(param, self.nu)
            penalized_loss = loss + self.lambda_ * penalty
        else:
            penalized_loss = loss

        return penalized_loss, loss

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between input and target.

        :param input: The model predictions.
        :type input: torch.Tensor
        :param target: The ground truth targets.
        :type target: torch.Tensor
        :return: The computed loss.
        :rtype: torch.Tensor
        :raises NotImplementedError: Subclasses must implement this method.
        """
        raise NotImplementedError("Subclasses must implement the 'compute_loss' method.")


class RegressionLoss(PenalizedLoss):
    """
    Custom loss function with an added penalty term for regression tasks :

    ..math:: \left\|\mathbf{y}-\mu_{\boldsymbol{\theta}}(X)\right\|_2 + \lambda P_\nu(\boldsymbol{\theta}^{(1)})

    where :math: P_\nu(\boldsymbol{\theta}^{(1)}) is the 'custom_penalty' function.

    :param lambda_: The regularization coefficient for the penalty term.
    :type lambda_: torch.Tensor
    :param nu: The exponent parameter for the non-convex penalty.
    :type nu: Optional[float]
    """
    def __init__(self, lambda_: torch.Tensor, nu: Optional[float]):
        super(RegressionLoss, self).__init__(lambda_, nu)
        self.criterion = nn.MSELoss(reduction='sum')

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = torch.sqrt(self.criterion(input, target))
        return loss


class ClassificationLoss(PenalizedLoss):
    """
    Custom loss function with an added penalty term for classification tasks:

    ..math:: \sum_{i=1}^n \mathbf{y}_i^{\top} \log \mu_\theta\left(\mathbf{x}_i\right) + \lambda P_\nu(\boldsymbol{\theta}^{(1)})

    :param lambda_: The regularization coefficient for the penalty term.
    :type lambda_: torch.Tensor
    :param nu: The exponent parameter for the non-convex penalty.
    :type nu: Optional[float]
    """
    def __init__(self, lambda_: torch.Tensor, nu: Optional[float]):
        super(ClassificationLoss, self).__init__(lambda_, nu)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')

    def compute_loss(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(input, target)
        return loss

class CoxLoss(PenalizedLoss):
    """
    Custom loss function for survival analysis using the Cox proportional hazards model.

    Assumes input and target are already sorted in descending time order.

    :param lambda_: The regularization coefficient for the penalty term.
    :type lambda_: torch.Tensor
    :param nu: The exponent parameter for the non-convex penalty.
    :type nu: Optional[float]
    """
    def __init__(self, lambda_: torch.Tensor, nu: Optional[float]):
        super(CoxLoss, self).__init__(lambda_, nu)

    def compute_loss(self, input, target, eps: float = 1e-7):
        events = target

        gamma = input.max()
        #log_cumsum_h = input.sub(gamma).exp().flip(0).cumsum(0).flip(0).add(eps).log().add(gamma)
        log_cumsum_h = input.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        return - input.sub(log_cumsum_h).mul(events).sum()
