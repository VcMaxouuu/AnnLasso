"""
Defines the CoxModel class for survival analysis tasks.
This class extends the BaseModel to implement functionality specific to Cox proportional hazards models,
including data preprocessing, baseline survival function computation, and survival function prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from ... import utils
from ... import core
from .base_model import BaseModel

class CoxModel(BaseModel):
    """
    Cox proportional hazards model for survival analysis tasks.

    :param hidden_dims: Dimensions of the hidden layers. If None, a linear model is trained.
    :type hidden_dims: Optional[Tuple[int, ...]]
    :param nu: Non-convexity parameter.
    :type nu: float
    :param lambda_qut: Lambda QUT value. If None, it will be computed during fitting.
    :type lambda_qut: Optional[torch.Tensor]
    """
    def __init__(
        self,
        hidden_dims: Optional[Tuple[int, ...]],
        nu: float,
        lambda_qut: Optional[torch.Tensor]
    ):
        super(CoxModel, self).__init__(hidden_dims, nu, lambda_qut)


    def _process_forward(self, output: torch.Tensor) -> torch.Tensor:
        """
        Process the forward pass for survival analysis.

        :param output: Output tensor from the model.
        :type output: torch.Tensor
        :return: The squeezed output tensor.
        :rtype: torch.Tensor
        """
        return output.squeeze()

    def _process_prediction(self, output: torch.Tensor) -> np.ndarray:
        """
        Convert the model's output tensor into predicted hazard ratios.

        :param output: Output tensor from the forward pass.
        :type output: torch.Tensor
        :return: Predicted hazard ratios.
        :rtype: np.ndarray
        """
        return output.cpu().numpy()

    def _preprocess_data(self, X: ArrayLike, durations_events: Tuple[ArrayLike, ArrayLike]):
        """
        Preprocess the input data for survival analysis.

        Converts the data into tensors, standardizes the input features, and sorts by durations.

        :param X: Input features, array-like of shape ``(n_samples, n_features)``.
        :type X: ArrayLike
        :param durations_events: Tuple containing durations and events.
        :type durations_events: Tuple[ArrayLike, ArrayLike]
        :return: Preprocessed input features and event indicators.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # Convert to tensors
        X = utils.X_to_tensor(X)
        durations, events = durations_events
        durations = utils.target_to_tensor(durations)
        events = utils.target_to_tensor(events)

        # Standardize the input features
        self.scaler = utils.StandardScaler()
        X = self.scaler.fit_transform(X)

        # Sort based on the durations
        _, idx = torch.sort(durations, descending=True)
        events = events[idx]
        X = X[idx]

        return X, events

    def fit(self, X, durations, events, verbose=False):
        """
        Override the 'BaseModel.fit' method for a method signature more specific to survival analysis
        and add the baseline survival function computation after fitting.

        :param X: Training data of shape ``(n_samples, n_features)``.
        :type X: ArrayLike
        :param durations: Observed times until event or censoring.
        :type durations: ArrayLike
        :param events: Event indicators (1 if event occurred, 0 if censored).
        :type events: ArrayLike
        :param verbose: Verbosity mode. Defaults to ``False``.
        :type verbose: bool
        """
        super().fit(X, (durations, events), verbose)
        self.baseline_surv = utils.compute_baseline_survival_function(self.predict(X), durations, events)

    def _build_layers(self, input_dim, output_dim, hidden_dims, device):
        """
        Build the layers of the Cox model.
        It has no intercept !

        :param input_dim: Number of input features.
        :type input_dim: int
        :param output_dim: Number of output features (always 1 for Cox model).
        :type output_dim: int
        :param hidden_dims: Dimensions of the hidden layers.
        :type hidden_dims: Tuple[int, ...]
        :param device: Device on which the layers will be created.
        :type device: torch.device
        :return: A list of model layers.
        :rtype: nn.ModuleList
        """
        return core.build_layers(input_dim, output_dim, hidden_dims, intercept=False, device=device)

    def _initialize_parameters(self, target):
        """
        Initialize the model parameters using a uniform distribution.

        :param target: Target values.
        :type target: torch.Tensor
        """
        core.initialize_uniform_parameters(self.named_parameters())

    def _get_output_dim(self, target) -> int:
        """
        Get the number of output dimensions (always 1 for Cox model).

        :param target: Target values.
        :type target: torch.Tensor
        :return: Number of output dimensions.
        :rtype: int
        """
        return 1

    def _compute_lambda_qut(self, X, target):
        """
        Compute the lambda QUT value for the Cox model.

        :param X: Input features.
        :type X: torch.Tensor
        :param target: Target values.
        :type target: torch.Tensor
        """
        if self.lambda_qut is None:
            self.lambda_qut = utils.lambda_qut_cox(
                X, target, self.act_fun, self.hidden_dims)
        if not torch.is_tensor(self.lambda_qut):
            self.lambda_qut = torch.tensor(self.lambda_qut)

    def _training_criterion(self, lambda_: float, nu: float) -> nn.Module:
        """
        Define the training criterion (loss function) for survival analysis.

        :param lambda_: Regularization parameter.
        :type lambda_: float
        :param nu: Non-convexity parameter.
        :type nu: float
        :return: The Cox loss function.
        :rtype: nn.Module
        """
        return utils.CoxLoss(lambda_, nu)

    def survival_function(self, X):
        """
        Compute the survival function for the given input data.

        :param X: Input features, array-like of shape ``(n_samples, n_features)``.
        :type X: ArrayLike
        :return: Survival probabilities for each individual.
        :rtype: np.ndarray
        """
        predictions = self.predict(X)
        baseline_survival = self.baseline_surv
        return utils.compute_individual_survival_functions(baseline_survival, predictions)

    def plot_survival_functions(self, X):
        """
        Plot the survival functions for the given input data.

        :param X: Input features, array-like of shape ``(n_samples, n_features)``.
        :type X: ArrayLike
        """
        survival_functions = self.survival_function(X)
        utils.plot_survival_functions(survival_functions)

    def plot_baseline_survival_function(self):
        """
        Plot the baseline survival function.
        """
        utils.plot_baseline_survival(self.baseline_surv)
