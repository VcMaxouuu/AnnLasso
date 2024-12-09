"""
Defines the BaseModel class for constructing harderLASSO models.
This base class provides a framework for feature selection and regularized training of models.
The class supports custom loss functions with their lambda qut and even optional hidden layers (for linear models)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from numpy.typing import ArrayLike
from ...utils import X_to_tensor
from ... import core

class BaseModel(nn.Module):
    """
    Base class for harderLASSO model construction.

    :param hidden_dims: Dimensions of the hidden layers. If None, a linear model is trained.
    :type hidden_dims: Optional[Tuple[int, ...]]
    :param nu: Non-convexity parameter.
    :type nu: float
    :param lambda_qut: Lambda QUT value. If None, it will be computed inside the 'fit' method.
    :type lambda_qut: Optional[torch.Tensor]
    """
    def __init__(
        self,
        hidden_dims: Optional[Tuple[int, ...]],
        nu: float,
        lambda_qut: Optional[torch.Tensor]
    ):
        """Constructor method
        """
        super(BaseModel, self).__init__()
        self.nu = nu
        self.hidden_dims = hidden_dims
        self.lambda_qut = lambda_qut
        self.act_fun = nn.ELU(alpha=1) if hidden_dims is not None else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Parameters to be determined during training
        self.imp_feat = None    # Important features after training
        self.scaler = None      # Data scaler for input normalization
        self.layers = None      # Neural network layers to be built during fitting


    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param X: Input data tensor.
        :type X: torch.Tensor
        :return: The output of the model.
        :rtype: torch.Tensor
        """

        output = X
        for i, layer in enumerate(self.layers):
            output = layer(output)
            if i < len(self.layers) - 1:
                output = self.act_fun(output)
        return self._process_forward(output)

    def _process_forward(self, output: torch.Tensor) -> torch.Tensor:
        """
        Process forward pass according to the task. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement '_process_output' method.")


    def predict(self, X: ArrayLike) -> np.ndarray:
        """
        Predict using the trained model.

        :param X: Test samples, array-like of shape ``(n_samples, n_features)``.
        :type X: ArrayLike
        :return: Predicted values or class labels.
        :rtype: np.ndarray
        """
        self.eval()
        X = X_to_tensor(X)
        X = self.scaler.transform(X)
        with torch.no_grad():
            output = self.forward(X[:, self.imp_feat[1]])
        return self._process_prediction(output)

    def _process_prediction(self, output: torch.Tensor) -> np.ndarray:
        """
        Process prediction according to the task. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement '_process_prediction' method.")

    def fit(self, X: ArrayLike, target: ArrayLike, verbose: bool = False):
        """
        Fit the model to the data.

        :param X: Training data.
        :type X: ArrayLike
        :param target: Target values.
        :type target: ArrayLike
        :param verbose: Verbosity mode. Defaults to ``False``.
        :type verbose: bool
        """
        ### Prepare Training ###
        # ----------------------
        # 1. Process data
        # 2. Create model layers
        # 3. Compute lambda_qut
        # ----------------------
        X, target = self._preprocess_data(X, target)
        self.layers = self._build_layers(X.shape[1], self._get_output_dim(target), self.hidden_dims, device=self.device)
        self._get_parameters_types()
        self._initialize_parameters(target)
        self._compute_lambda_qut(X, target)

        ### Training Phase ###
        # -----------------
        # 1. Training without regularization
        # 2. Training with regularization
        # 3. Pruning model first two layers
        # 4. Training without regularization
        # -----------------
        core.perform_Adam(self, X, target, 0.01, torch.tensor(0), self.nu, 1e-5, self._training_criterion, verbose)
        core.training_loop(self, X, target, self.nu, self.lambda_qut, self._training_criterion, verbose)
        self.imp_feat = core.extract_important_features(self.layers[0].weight.data)
        self._pruning()
        core.perform_Adam(self, X[:, self.imp_feat[1]], target, 0.01, torch.tensor(0), self.nu, 1e-10, self._training_criterion, verbose)

    def _preprocess_data(self, X: ArrayLike, target: ArrayLike):
        """
        Preprocess data. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement '_preprocess_data' method.")

    def _get_parameters_types(self):
        """
        Split the parameters into the penalized and unpenalized ones.
        """
        self.penalized_parameters = core.get_penalized_parameters(self)
        self.unpenalized_parameters = core.get_unpenalized_parameters(self)

    def _build_layers(self, input_dim: int, output_dim: int, hidden_dims: Tuple[int,...], device: torch.device) -> nn.ModuleList:
        """
        Build the model layers. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement '_build_layers' method.")

    def _initialize_parameters(self, target):
        """
        Initialize model parameters. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement '_initialize_parameters' method.")

    def _get_output_dim(self, target) -> int:
        """
        Get the output dimension. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement '_get_output_dim' method.")

    def _compute_lambda_qut(self, X, target):
        """
        Compute lambda_qut. To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement '_compute_lambda_qut' method.")

    def _pruning(self):
        """
        Prune the model based on the important features.
        """
        if self.hidden_dims is not None:
            active_neurons = torch.nonzero(~torch.all(self.layers[0].weight == 0, dim=1), as_tuple=True)[0]

            pruned_weight_0 = self.layers[0].weight.data[active_neurons, :][:, self.imp_feat[1]]
            pruned_bias_0 = self.layers[0].bias.data[active_neurons]
            self.layers[0].weight = nn.Parameter(pruned_weight_0)
            if self.layers[0].bias is not None:
                self.layers[0].bias = nn.Parameter(pruned_bias_0)

            self.layers[0].in_features = pruned_weight_0.shape[1]
            self.layers[0].out_features = pruned_weight_0.shape[0]

            pruned_weight_1 = self.layers[1].weight.data[:, active_neurons]
            self.layers[1].weight = nn.Parameter(pruned_weight_1)
            self.layers[1].in_features = pruned_weight_1.shape[1]

            self.hidden_dims = (len(active_neurons),) + self.hidden_dims[1:]
        else:
            pruned_weight_0 = self.layers[0].weight.data[:, self.imp_feat[1]]
            self.layers[0].weight = nn.Parameter(pruned_weight_0)
            self.layers[0].in_features = pruned_weight_0.shape[1]
            self.layers[0].out_features = pruned_weight_0.shape[0]

        self._get_parameters_types()

    def _training_criterion(self) -> nn.Module:
        """
        Training criterion (cost function). To be implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement '_training_criterion' method.")

    def save(self, filepath: str):
        """
        Save the model to a file.

        :param filepath: File path to save the model. Should be of ``.pth`` format.
        :type filepath: str
        """
        save_dict = {'state_dict': self.state_dict()}
        additional_info = {
            'scaler': self.scaler,
            'lambda_qut': self.lambda_qut,
        }
        save_dict.update(additional_info)
        torch.save(save_dict, filepath)

    def load(self, filepath: str, strict: bool = True):
        """
        Load the model from the given file.

        :param filepath: Path to load the model from. Should be of ``.pth`` format.
        :type filepath: str
        :param strict: Whether to strictly enforce matching keys in 'state_dict'. Defaults to ``True``.
        :type strict: bool
        """
        checkpoint = torch.load(filepath, weights_only=False)
        self.scaler = checkpoint.get('scaler', None)
        self.lambda_qut = checkpoint.get('lambda_qut', None)

        state_dict = checkpoint['state_dict']

        dimensions = [state_dict['layers.0.weight'].shape[1]] + [param.shape[0] for name, param in state_dict.items() if 'weight' in name]
        self.layers = self._build_layers(dimensions[0], dimensions[-1], dimensions[1:-1], device=self.device)
        self.load_state_dict(state_dict, strict=strict)
        self._get_parameters_types()

        self.imp_feat = core.extract_important_features(self.layers[0].weight.data)
