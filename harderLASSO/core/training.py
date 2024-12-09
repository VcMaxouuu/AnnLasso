"""
Contains the main training loop and optimization logic.
Includes functions for monitoring and convergence checks.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Callable
from .optimization import ISTABlock
from .model_parameters import extract_important_features


def training_loop(
    model: nn.Module,
    X: torch.Tensor,
    target: torch.Tensor,
    nu: float,
    lambda_: torch.Tensor,
    criterion_function: Callable,
    verbose: bool
) -> None:
    """
    Main training loop that starts with ADAM phases and a final FISTA phase
    for penalized optimization.

    :param model: The neural network model to be trained.
    :type model: nn.Module
    :param X: Training data of shape (n_samples, n_features).
    :type X: torch.Tensor
    :param target: Target values of shape (n_samples,).
    :type target: torch.Tensor
    :param nu: Non-convexity parameter.
    :type nu: float
    :param lambda_: Regularization parameter.
    :type lambda_: torch.Tensor
    :param criterion_function: Function to compute the loss.
    :type criterion_function: Callable
    :param verbose: If True, print detailed logs during training.
    :type verbose: bool
    """

    # Ensure model is in training mode
    model.train()

    # Define the different parameters for each phases,
    # should converge to the final optimal values
    if nu is None:
        nu_schedule = [None] * 6
    else:
        nu_schedule = [1, 0.7, 0.4, 0.3, 0.2, 0.1]
    lambda_schedule = [np.exp(i - 1) / (1 + np.exp(i - 1)) for i in range(5)] + [1]

    for i in range(6):
        lambda_i = lambda_schedule[i] * lambda_
        nu_value = nu_schedule[i]

        learning_rate = 0.01
        rel_err = 1e-6

        # One could also perform ISTA for each phase, but that would require longer training time
        if i < 5:
            if verbose:
                print(f"### Intermediate phase {i + 1}: Lambda = {lambda_i.item():.4f}" +
                    (f", Nu = {nu_value}" if nu_value is not None else "") + " ###")
            perform_Adam(
                model, X, target, learning_rate, lambda_i, nu_value, rel_err, criterion_function, verbose=verbose
            )
        else :
            if verbose:
                print("### Final ISTA Phase ###")
            perform_ISTA(
                model, X, target, learning_rate, lambda_i, nu_value, rel_err, criterion_function, verbose=verbose
            )


def perform_ISTA(
    model: nn.Module,
    X: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
    lambda_: torch.Tensor,
    nu: float,
    rel_err: float,
    criterion_function: Callable,
    verbose: bool,
    logging_interval: int = 50
) -> None:
    """
    Train the model using ISTA with penalized optimization.

    Descent stops when the relative error is below the `rel_err` threshold.

    :param model: The neural network model to be trained.
        Should have a 'penalized_parameters' attribute.
    :type model: nn.Module
    :param X: Training data of shape (n_samples, n_features).
    :type X: torch.Tensor
    :param target: Target values of shape (n_samples,).
    :type target: torch.Tensor
    :param learning_rate: Learning rate for the FISTABlock optimizer.
        If an Armijo rule is used in the optimizer, this corresponds to the maximum learning rate.
    :type learning_rate: float
    :param lambda_: Regularization parameter.
    :type lambda_: torch.Tensor
    :param nu: Non-convexity parameter.
    :type nu: float
    :param rel_err: Convergence threshold for relative error.
    :type rel_err: float
    :param criterion_function: Function to compute the loss.
        Should be a subclass of `nn.Module` with `forward` and `backward` methods.
    :type criterion_function: Callable
    :param verbose: If True, print training logs.
    :type verbose: bool
    :param logging_interval: Interval at which the logs are printed if verbose, defaults to 50.
    :type logging_interval: int, optional
    """

    penalized_weights = [p for name, p in model.penalized_parameters if 'weight' in name]
    penalized_biases = [p for name, p in model.penalized_parameters if 'bias' in name]
    unpenalized_params = [p for _, p in model.unpenalized_parameters]

    optimizer =  ISTABlock(
                                [
                                    {'params': penalized_biases, 'lambda_': lambda_},
                                    {'params': penalized_weights, 'lambda_': lambda_},
                                    {'params': unpenalized_params, 'lambda_': torch.tensor(0.0)}
                                ],
                                lr=learning_rate, lambda_=lambda_, nu=nu
                            )


    criterion = criterion_function(lambda_, nu)

    epoch, last_loss = 0, torch.tensor(np.inf)

    def closure(backward):
        optimizer.zero_grad()
        predictions = model(X)
        loss, bare_loss = criterion(predictions, target, model.penalized_parameters)
        if backward:
            bare_loss.backward()
        return loss, bare_loss

    while True:
        loss, bare_loss = optimizer.step(closure)

        if verbose and epoch % logging_interval == 0:
            important_features = extract_important_features(model.layers[0].weight.data)
            print(
                f"\t Epoch {epoch}: Loss = {loss.item():.5f}, Mean Bare Loss = {bare_loss.item()/X.shape[0]:.5f}, "
                f"Important Features = {important_features}"
            )

        if check_convergence(loss, last_loss, rel_err):
            if verbose:
                print(f"\t\tConverged after {epoch} epochs. Relative loss change below {rel_err}.\n")
            break

        last_loss = loss
        epoch += 1


def perform_Adam(
    model: nn.Module,
    X: torch.Tensor,
    target: torch.Tensor,
    learning_rate: float,
    lambda_: torch.Tensor,
    nu: float,
    rel_err: float,
    criterion_function: Callable,
    verbose: bool,
    logging_interval: int = 50
) -> None:
    """
    Train the model using the Adam optimizer with penalized optimization.

    Descent stops when the relative error is below the `rel_err` threshold.

    :param model: The neural network model to be trained.
        Should have a 'penalized_parameters' attribute.
    :type model: nn.Module
    :param X: Training data of shape (n_samples, n_features).
    :type X: torch.Tensor
    :param target: Target values of shape (n_samples,).
    :type target: torch.Tensor
    :param learning_rate: Learning rate for the Adam optimizer.
        The optimizer uses a scheduler to reduce the learning rate if the loss does not decrease.
    :type learning_rate: float
    :param lambda_: Regularization parameter.
    :type lambda_: torch.Tensor
    :param nu: Non-convexity parameter.
    :type nu: float
    :param rel_err: Convergence threshold for relative error.
    :type rel_err: float
    :param criterion_function: Function to compute the loss.
        Should be a subclass of `nn.Module` with `forward` and `backward` methods.
    :type criterion_function: Callable
    :param verbose: If True, print training logs.
    :type verbose: bool
    :param logging_interval: Interval at which the logs are printed if verbose, defaults to 50.
    :type logging_interval: int, optional
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=10, factor=0.5, min_lr=1e-6
    )
    criterion = criterion_function(lambda_, nu)

    epoch, last_loss = 0, torch.tensor(np.inf)

    while True:
        optimizer.zero_grad()
        predictions = model(X)
        loss, bare_loss = criterion(predictions, target, model.penalized_parameters)

        if verbose and epoch % logging_interval == 0:
            print(
                f"\tEpoch {epoch}: Total Loss = {loss.item():.5f}, "
                f"Mean Bare Loss = {bare_loss.item()/X.shape[0]:.5f}"
            )

        if check_convergence(loss, last_loss, rel_err):
            if verbose:
                print(f"\t\tConverged after {epoch} epochs. Relative loss change below {rel_err}.\n")
            break

        if loss <= 0.1:
            if verbose:
                print(f"\t\tConverged after {epoch} epochs. Penalized loss is small.\n")
            break

        last_loss = loss

        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        epoch += 1


def check_convergence(current_loss: torch.Tensor, last_loss: torch.Tensor, rel_err: float) -> bool:
    """
    Check whether the relative change in loss is below the convergence threshold.

    :param current_loss: Current loss value.
    :type current_loss: torch.Tensor
    :param last_loss: Last recorded loss value.
    :type last_loss: torch.Tensor
    :param rel_err: Convergence threshold for relative error.
    :type rel_err: float
    :return: True if converged, False otherwise.
    :rtype: bool
    """

    if torch.isinf(last_loss) or last_loss == 0:
        return False
    relative_change = torch.abs(current_loss - last_loss) / torch.abs(last_loss)
    return relative_change < rel_err
