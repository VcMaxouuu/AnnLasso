"""
Defines the ClassifierModel class for classification tasks.
This class extends the BaseModel to implement functionality specific to classification, including
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

class ClassifierModel(BaseModel):
    """
    Classifier model for classification tasks.

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
        super(ClassifierModel, self).__init__(hidden_dims, nu, lambda_qut)


    def _process_forward(self, output: torch.Tensor) -> torch.Tensor:
        """
        Process the forward pass for classification.

        :param output: Output tensor from the model.
        :type output: torch.Tensor
        :return: The output tensor (logits).
        :rtype: torch.Tensor
        """
        return output

    def _process_prediction(self, output: torch.Tensor) -> np.ndarray:
        """
        Convert the model's output logits into predicted class labels.

        :param output: Output tensor from the forward pass.
        :type output: torch.Tensor
        :return: Predicted class labels.
        :rtype: np.ndarray
        """
        return torch.argmax(output, dim=1).cpu().numpy()

    def _preprocess_data(self, X: ArrayLike, target: ArrayLike):
        """
        Preprocess the input data for classification.

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
        target = utils.target_to_tensor(target).long()

        # Standardize the input features
        self.scaler = utils.StandardScaler()
        X = self.scaler.fit_transform(X)

        return X, target

    def _build_layers(self, input_dim, output_dim, hidden_dims, device):
        """
        Build the layers of the classification model.

        :param input_dim: Number of input features.
        :type input_dim: int
        :param output_dim: Number of output classes.
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
        Initialize the model parameters using a uniform distribution.

        :param target: Target values.
        :type target: torch.Tensor
        """
        core.initialize_uniform_parameters(self.parameters())

    def _get_output_dim(self, target) -> int:
        """
        Get the number of unique classes in the target data.

        :param target: Target values.
        :type target: torch.Tensor
        :return: Number of unique classes.
        :rtype: int
        """
        return len(torch.unique(target))

    def _compute_lambda_qut(self, X, target):
        """
        Compute the lambda QUT value for the classification task.

        :param X: Input features.
        :type X: torch.Tensor
        :param target: Target values.
        :type target: torch.Tensor
        """
        if self.lambda_qut is None:
            self.lambda_qut = utils.lambda_qut_classification(
                X, utils.get_hat_p(target), self.act_fun, self.hidden_dims)
        if not torch.is_tensor(self.lambda_qut):
            self.lambda_qut = torch.tensor(self.lambda_qut)

    def _training_criterion(self, lambda_: float, nu: float) -> nn.Module:
        """
        Define the training criterion (loss function) for classification.

        :param lambda_: Regularization parameter.
        :type lambda_: float
        :param nu: Non-convexity parameter.
        :type nu: float
        :return: The classification loss function.
        :rtype: nn.Module
        """
        return utils.ClassificationLoss(lambda_, nu)
