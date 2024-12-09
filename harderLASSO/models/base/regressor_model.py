"""
Defines the RegressionModel class for regression tasks.
This class extends the BaseModel to implement functionality specific to regression, including
data preprocessing, layer construction, parameter initialization, and lambda QUT computation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from ... import utils
from ... import core
from .base_model import BaseModel

class RegressionModel(BaseModel):
    """
    Regression model for regression tasks.

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
        super(RegressionModel, self).__init__(hidden_dims, nu, lambda_qut)


    def _process_forward(self, output: torch.Tensor) -> torch.Tensor:
        """
        Process the forward pass for regression.

        :param output: Output tensor from the model.
        :type output: torch.Tensor
        :return: The squeezed output tensor.
        :rtype: torch.Tensor
        """
        return output.squeeze()

    def _process_prediction(self, output: torch.Tensor) -> np.ndarray:
        """
        Convert the model's output tensor into predicted values.

        :param output: Output tensor from the forward pass.
        :type output: torch.Tensor
        :return: Predicted values.
        :rtype: np.ndarray
        """
        return output.cpu().numpy()

    def _preprocess_data(self, X: ArrayLike, target: ArrayLike):
        """
        Preprocess the input data for regression.

        Converts the data into tensors and standardizes the input features.

        :param X: Input features, array-like of shape ``(n_samples, n_features)``.
        :type X: ArrayLike
        :param target: Target values, array-like of shape ``(n_samples,)``.
        :type target: ArrayLike
        :return: Tuple of preprocessed input features and target values.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        # Convert to tensors
        X = utils.X_to_tensor(X)
        target = utils.target_to_tensor(target)

        # Standardize the input features
        self.scaler = utils.StandardScaler()
        X = self.scaler.fit_transform(X)

        return X, target

    def _build_layers(self, input_dim, output_dim, hidden_dims, device):
        """
        Build the layers of the regression model.

        :param input_dim: Number of input features.
        :type input_dim: int
        :param output_dim: Number of output features (always 1 for regression).
        :type output_dim: int
        :param hidden_dims: Dimensions of the hidden layers.
        :type hidden_dims: Tuple[int, ...]
        :param device: Device on which the layers will be created.
        :type device: torch.device
        :return: A list of model layers.
        :rtype: nn.ModuleList
        """
        return core.build_layers(input_dim, output_dim, hidden_dims, intercept=True, device=device)

    def _initialize_parameters(self, target):
        """
        Initialize the model parameters.

        Penalized parameters are initialized with a normal distribution, while
        unpenalized parameters are initialized uniformly.

        :param target: Target values used to compute standard deviation for initialization.
        :type target: torch.Tensor
        """
        core.initialize_normal_parameters([param for _, param in self.penalized_parameters], mean=0.0, std=target.std())
        core.initialize_uniform_parameters([param for _, param in self.unpenalized_parameters])

    def _get_output_dim(self, target) -> int:
        """
        Get the number of output dimensions (always 1 for regression).

        :param target: Target values.
        :type target: torch.Tensor
        :return: Number of output dimensions.
        :rtype: int
        """
        return 1

    def _compute_lambda_qut(self, X, target):
        """
        Compute the lambda QUT value for the regression task.

        :param X: Input features.
        :type X: torch.Tensor
        :param target: Target values.
        :type target: torch.Tensor
        """
        if self.lambda_qut is None:
            self.lambda_qut = utils.lambda_qut_regression(
                X, self.act_fun, self.hidden_dims)
        if not torch.is_tensor(self.lambda_qut):
            self.lambda_qut = torch.tensor(self.lambda_qut)

    def _training_criterion(self, lambda_: float, nu: float) -> nn.Module:
        """
        Define the training criterion (loss function) for regression.

        :param lambda_: Regularization parameter.
        :type lambda_: float
        :param nu: Non-convexity parameter.
        :type nu: float
        :return: The regression loss function.
        :rtype: nn.Module
        """
        return utils.RegressionLoss(lambda_, nu)
