"""
Implements the shrinkage operator and related functions used
for regularization in optimization processes.
"""

import torch
import numpy as np
import warnings
from scipy.optimize import root_scalar, newton

def shrinkage_operator(u: torch.Tensor, lambda_: torch.Tensor, nu: float) -> torch.Tensor:
    """
    Apply the shrinkage operator to the input tensor `u`.

    :param u: Input tensor to be shrinked.
    :type u: torch.Tensor
    :param lambda_: Regularization parameter. Should be a scalar tensor.
    :type lambda_: torch.Tensor
    :param nu: Non-convexity parameter. Should be either ``None`` or a float smaller than zero.
    :type nu: float
    :return: The resulting tensor after applying the shrinkage operator.
    :rtype: torch.Tensor
    """
    # If regularization is zero, no shrinkage is applied.
    if not torch.is_nonzero(lambda_):
        return u

    # If nu is None, perfom basic soft tresholding.
    if nu is None:
        return torch.sign(u) * torch.max(torch.zeros_like(u), u.abs() - lambda_)

    # If nu is not None, find treshold and solve corresponding equation.
    phi = find_thresh(nu, lambda_.item())
    sol = torch.zeros_like(u)
    abs_u = u.abs()
    ind = abs_u > phi

    if not ind.any():
        return sol

    sol_values = torch.tensor(
        nonzerosol(u[ind].cpu().numpy(), nu, lambda_.item()),
        dtype=u.dtype,
        device=u.device
    )
    sol[ind] = sol_values
    return sol


def find_thresh(nu: float, lambda_: float) -> float:
    """
    Compute the threshold value for the shrinkage operator.

    Utilizes SciPy optimizations; hence parameters cannot be tensors.

    :param nu: Non-convexity parameter.
    :type nu: float
    :param lambda_: Regularization parameter.
    :type lambda_: float
    :return: The calculated threshold value `phi`.
    :rtype: float
    """

    def func(kappa):
        return kappa**(2 - nu) + 2 * kappa + kappa**nu + 2 * lambda_ * (nu - 1)

    bracket = [0, lambda_ * (1 - nu) / 2]
    solution = root_scalar(func, bracket=bracket)
    kappa = solution.root
    phi = kappa / 2 + lambda_ / (1 + kappa**(1 - nu))
    return phi

def nonzerosol(u: np.ndarray, nu: float, lambda_: float) -> np.ndarray:
    """
    Compute the nonzero solution of the non-convex penalty function.

    Utilizes SciPy optimizations; hence parameters cannot be tensors.

    :param u: Input array of values.
    :type u: np.ndarray
    :param nu: Non-convexity parameter.
    :type nu: float
    :param lambda_: Regularization parameter.
    :type lambda_: float
    :return: Array of solutions.
    :rtype: np.ndarray

    :notes:
        If the method does not converge, it returns the absolute value of `u`.
    """

    def func(x, y, lambda_, nu):
        return x - np.abs(y) + lambda_ * (1 + nu * x**(1 - nu)) / (1 + x**(1 - nu))**2

    def fprime(x, y, lambda_, nu):
        numerator = (1 - nu) * (2 + nu * x**(1 - nu) - nu)
        denominator = x**nu * (1 + x**(1 - nu))**3
        return 1 - lambda_ * (numerator / denominator)

    try:
        root = newton(
            func,
            x0=np.abs(u),
            fprime=fprime,
            args=(u, lambda_, nu),
            maxiter=500,
            tol=1e-5
        )
    except RuntimeError:
        warnings.warn(
            "Newton-Raphson did not converge. Returning absolute value of input.",
            RuntimeWarning
        )
        root = np.abs(u)
    return root * np.sign(u)
