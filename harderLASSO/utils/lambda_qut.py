"""
Provides utilities for computing Quantile Universal Thresholds (QUT) for various tasks
such as regression, classification, and survival analysis.
"""

import torch
import numpy as np
from typing import Callable, Optional, Tuple

def function_derivative(func: Callable[[torch.Tensor], torch.Tensor], u: torch.Tensor) -> float:
    """
    Computes the derivative of a function `func` at point `u` using PyTorch autograd.

    :param func: A differentiable function that takes a PyTorch tensor and returns a scalar tensor.
    :type func: Callable[[torch.Tensor], torch.Tensor]
    :param u: A tensor at which to compute the derivative. Must have `requires_grad=True`.
    :type u: torch.Tensor
    :return: The derivative of `func` at `u`.
    :rtype: float
    :raises ValueError: If `u` does not have `requires_grad=True`.
    """
    if not u.requires_grad:
        raise ValueError("Input tensor `u` must have `requires_grad=True`.")
    if u.grad is not None:
        u.grad.zero_()
    y = func(u)
    y.backward()
    derivative = u.grad.item()
    u.grad.zero_()
    return derivative

def _apply_activation_scaling(
    full_list: torch.Tensor,
    act_fun: Callable[[torch.Tensor], torch.Tensor] = None,
    hidden_dims: Tuple[int, ...] = (20, ),
    alpha: float = 0.05,
) -> torch.Tensor:
    """
    Applies activation function scaling to the simulated statistics.
    If 'act_fun' is None, the corresponding model is linear hence no scaling is applied.

    :param full_list: The tensor containing simulated statistics.
    :type full_list: torch.Tensor
    :param act_fun: Activation function. Defaults to ``None``.
    :type act_fun: Callable[[torch.Tensor], torch.Tensor], optional
    :param hidden_dims: Dimensions of hidden layers. Defaults to ``(20,)``.
    :type hidden_dims: Tuple[int, ...], optional
    :param alpha: Significance level. Defaults to ``0.05``.
    :type alpha: float
    :return: The scaled tensor.
    :rtype: torch.Tensor
    """
    if act_fun is None:
        return torch.quantile(full_list, 1 - alpha)

    if len(hidden_dims) == 1:
        pi_l = 1.0
    else:
        pi_l = np.sqrt(np.prod(hidden_dims[1:]))

    u = torch.tensor(0.0, dtype=torch.float32, device=full_list.device, requires_grad=True)
    sigma_diff = function_derivative(act_fun, u) ** len(hidden_dims)
    full_list = full_list * pi_l * sigma_diff
    return torch.quantile(full_list, 1 - alpha)

def _lambda_qut(
    simulate_func: Callable[[int], float],
    device: torch.device,
    act_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hidden_dims: Tuple[int, ...] = (20,),
    n_samples: int = 5000,
    alpha: float = 0.05
) -> torch.Tensor:
    """
    Generalized function to compute the quantile universal threshold (QUT).

    :param simulate_func: A function that runs a single simulation and returns a statistic.
    :type simulate_func: Callable[[int], float]
    :param device: The device on which to run the simulations.
    :type device: torch.device
    :param act_fun: The activation function used in the neural network. If ``None``, no scaling is applied (linear model). Defaults to ``None``.
    :type act_fun: Optional[Callable[[torch.Tensor], torch.Tensor]], optional
    :param hidden_dims: The dimensions of the hidden layers in the neural network. Defaults to ``(20,)``.
    :type hidden_dims: Tuple[int, ...], optional
    :param n_samples: The number of simulations to run. Defaults to ``5000``.
    :type n_samples: int, optional
    :param alpha: The significance level for the quantile. Defaults to ``0.05``.
    :type alpha: float, optional
    :return: The computed quantile value.
    :rtype: torch.Tensor
    """
    full_list = torch.zeros(n_samples, device = device)

    for i in range(n_samples):
        full_list[i] = simulate_func()

    return _apply_activation_scaling(full_list, act_fun, hidden_dims, alpha)


def lambda_qut_regression(
    X: torch.Tensor,
    act_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hidden_dims: Tuple[int, ...] = (20,),
    n_samples: int = 5000,
    alpha: float = 0.05
) -> torch.Tensor:
    """
    Computes the quantile universal threshold (QUT) for regression tasks.

    :param X: The design matrix of shape ``(n_samples, n_features)``.
    :type X: torch.Tensor
    :param act_fun: The activation function used in the neural network. If ``None``, linear regression is assumed. Defaults to ``None``.
    :type act_fun: Optional[Callable[[torch.Tensor], torch.Tensor]], optional
    :param hidden_dims: The dimensions of the hidden layers in the neural network. Defaults to ``(20,)``.
    :type hidden_dims: Tuple[int, ...], optional
    :param n_samples: The number of simulations to run. Defaults to ``5000``.
    :type n_samples: int, optional
    :param alpha: The significance level for the quantile. Defaults to ``0.05``.
    :type alpha: float, optional
    :return: The computed quantile value.
    :rtype: torch.Tensor
    """
    def simulate_regression():
        y_sample = torch.normal(mean=0., std=1, size=(X.shape[0], 1), device=X.device)
        y = y_sample - y_sample.mean()
        xy = torch.matmul(X.T, y)
        xy_sum = torch.abs(xy).sum(dim=1)
        xy_max = xy_sum.max()
        y_norm = y.norm(p=2)
        return xy_max / y_norm

    return _lambda_qut(
        simulate_func=simulate_regression,
        device=X.device,
        act_fun=act_fun,
        hidden_dims=hidden_dims,
        n_samples=n_samples,
        alpha=alpha
    )

def lambda_qut_classification(
    X: torch.Tensor,
    hat_p: torch.Tensor,
    act_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hidden_dims: Tuple[int, ...] = (20,),
    n_samples: int = 5000,
    alpha: float = 0.05
) -> torch.Tensor:
    """
    Computes the quantile universal threshold (QUT) for classification tasks.

    :param X: The design matrix of shape ``(n_samples, n_features)``.
    :type X: torch.Tensor
    :param hat_p: The estimated class probabilities of shape ``(n_classes,)``.
    :type hat_p: torch.Tensor
    :param act_fun: The activation function used in the neural network. If ``None``, linear classification is assumed. Defaults to ``None``.
    :type act_fun: Optional[Callable[[torch.Tensor], torch.Tensor]], optional
    :param hidden_dims: The dimensions of the hidden layers in the neural network. Defaults to ``(20,)``.
    :type hidden_dims: Tuple[int, ...], optional
    :param n_samples: The number of simulations to run. Defaults to ``5000``.
    :type n_samples: int, optional
    :param alpha: The significance level for the quantile. Defaults to ``0.05``.
    :type alpha: float, optional
    :return: The computed quantile value.
    :rtype: torch.Tensor
    """

    def simulate_classification():
        y_sample_indices = torch.multinomial(hat_p, X.shape[0], replacement=True).to(X.device)
        y_sample_one_hot = torch.nn.functional.one_hot(y_sample_indices, num_classes=hat_p.shape[0]).float()
        y = y_sample_one_hot - y_sample_one_hot.mean(dim=0)
        xy = torch.matmul(X.T, y)
        xy_sum = torch.abs(xy).sum(dim=1)
        xy_max = xy_sum.max()
        return xy_max

    return _lambda_qut(
        simulate_func=simulate_classification,
        device=X.device,
        act_fun=act_fun,
        hidden_dims=hidden_dims,
        n_samples=n_samples,
        alpha=alpha
    )

def lambda_qut_cox(
    X: torch.Tensor,
    c: torch.Tensor,
    act_fun: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    hidden_dims: Tuple[int, ...] = (20,),
    n_samples: int = 5000,
    alpha: float = 0.05
) -> torch.Tensor:
    """
    Computes the quantile universal threshold (QUT) for survival analysis tasks.

    :param X: The design matrix of shape ``(n_samples, n_features)``.
    :type X: torch.Tensor
    :param c: The tensor containing the censoring indicators of shape ``(n_samples,)``.
    :type c: torch.Tensor
    :param act_fun: The activation function used in the neural network. If ``None``, linear regression is assumed. Defaults to ``None``.
    :type act_fun: Optional[Callable[[torch.Tensor], torch.Tensor]], optional
    :param hidden_dims: The dimensions of the hidden layers in the neural network. Defaults to ``(20,)``.
    :type hidden_dims: Tuple[int, ...], optional
    :param n_samples: The number of simulations to run. Defaults to ``5000``.
    :type n_samples: int, optional
    :param alpha: The significance level for the quantile. Defaults to ``0.05``.
    :type alpha: float, optional
    :return: The computed quantile value.
    :rtype: torch.Tensor
    """
    def simulate_cox():
        n = X.shape[0]
        y_sample = torch.rand(n, )
        c_sample = c[torch.randperm(n)]

        _, sorted_indices = torch.sort(y_sample, descending=True)
        sorted_X = X[sorted_indices]

        sorted_c = c_sample[sorted_indices]
        event_indices = torch.where(sorted_c == 1)[0]

        risk = sorted_X
        risk_cumsums_mean =  risk.cumsum(0) / torch.arange(1, risk.shape[0]+1).unsqueeze(1)
        result = - (risk - risk_cumsums_mean)[event_indices].sum(dim=0)
        result = torch.max(torch.abs(result))
        return result

    return _lambda_qut(
        simulate_func=simulate_cox,
        device=X.device,
        act_fun=act_fun,
        hidden_dims=hidden_dims,
        n_samples=n_samples,
        alpha=alpha
    )
