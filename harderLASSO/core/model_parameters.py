"""
Provides utilities for building neural network layers,
initializing parameters, and managing penalized/unpenalized parameters.
"""
import torch.nn as nn
import torch
from typing import Optional, Tuple, List
from torch.nn.functional import normalize
from math import sqrt


def build_layers(input_dim: int, output_dim: int, hidden_dims: Optional[Tuple[int, ...]], intercept: bool, device: torch.device) -> nn.ModuleList:
    """
    Build the neural network layers given the needed dimensions.

    If no hidden dimensions are provided, a linear model will be created.
    If hidden dimensions are provided, the neural network will use normalized weights for layers after the first one.

    :param input_dim: The dimension of the input features.
    :type input_dim: int
    :param output_dim: The dimension of the output.
    :type output_dim: int
    :param hidden_dims: Dimensions of the hidden layers. If None or empty, a linear model is built.
    :type hidden_dims: Optional[Tuple[int, ...]]
    :param intercept: Whether to include an intercept term in the model (bias in the last layer).
    :type intercept: bool
    :param device: The device on which the parameters will be placed.
    :type device: torch.device
    :return: A list of neural network layers.
    :rtype: nn.ModuleList
    """
    layers = nn.ModuleList()

    if not hidden_dims:
        # Linear model
        layers.append(nn.Linear(input_dim, output_dim, bias=intercept, device=device))
    else:
        layers.append(nn.Linear(input_dim, hidden_dims[0], device=device))
        for i in range(1, len(hidden_dims)):
            layers.append(NormalizedLinear(hidden_dims[i - 1], hidden_dims[i], device=device))
        layers.append(NormalizedLinear(hidden_dims[-1], output_dim, bias=intercept, device=device))
    return layers


def initialize_normal_parameters(parameters: List[nn.Parameter], mean: float = 0.0, std: float = 1.0):
    """
    Initializes the given parameters with a normal distribution.

    :param parameters: List of parameters to initialize.
    :type parameters: List[nn.Parameter]
    :param mean: Mean of the normal distribution. Default is 0.0.
    :type mean: float
    :param std: Standard deviation of the normal distribution. Default is 1.0.
    :type std: float
    """
    for param in parameters:
        param.data.normal_(mean=mean, std=std)

def initialize_uniform_parameters(parameters: List[nn.Parameter]):
    """
    Initialize model parameters with a Kaiming He distribution.

    :param parameters: List of parameters to initialize.
    :type parameters: List[nn.Parameter]
    """
    for param in parameters:
        n = param.shape[1] if param.dim() > 1 else param.shape[0]
        stdv = 1.0/sqrt(n)
        param.data.uniform_(-stdv, stdv)


def get_penalized_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """
    Retrieve the parameters (name and state) of the given model that will be penalized during training.

    :param model: The neural network model.
    :type model: nn.Module
    :return: A list containing tuples of parameter names and parameters to be penalized.
    :rtype: List[Tuple[str, nn.Parameter]]
    """
    penalized_params = [('layers.0.weight', model.layers[0].weight)]
    for i, layer in enumerate(model.layers[:-1]):  # Exclude the last layer
        param_name = f'layers.{i}.bias'
        penalized_params.append((param_name, layer.bias))
    return penalized_params

def get_unpenalized_parameters(model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
    """
    Retrieve the parameters (name and state) of the given model that will not be penalized during training.

    Must be called after initializing the penalized parameters.

    :param model: The neural network model.
    :type model: nn.Module
    :return: A list containing tuples of parameter names and parameters not to be penalized.
    :rtype: List[Tuple[str, nn.Parameter]]
    """
    all_params = list(model.named_parameters())
    unpenalized_params = [(name, param) for name, param in all_params if name not in [pen_name for pen_name, _ in model.penalized_parameters]]

    return unpenalized_params


def extract_important_features(weight: torch.Tensor) -> Tuple[int, List[int]]:
    """
    Identify important features as the non-zero columns of a weight parameter.

    Should be called on the first layer weight of a trained neural network.

    :param weight: Weight parameter.
    :type weight: torch.Tensor
    :return: A tuple containing the count of important features and their indices.
    :rtype: Tuple[int, List[int]]
    """
    non_zero_columns = torch.any(weight != 0, dim=0)
    count = torch.count_nonzero(non_zero_columns).item()
    indices = torch.where(non_zero_columns)[0].tolist()
    return count, sorted(indices)

class NormalizedLinear(nn.Module):
    """
    Applies a normalized affine linear transformation to the incoming data.

    The transformation is defined as:
    .. math:: y = \\mathbf{b} + A^{\\circ} \\mathbf{x}

    where :math:`A^{\\circ}` denotes a L2 row-wise normalized version of `A`.

    :param in_features: Size of each input sample.
    :type in_features: int
    :param out_features: Size of each output sample.
    :type out_features: int
    :param bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``.
    :type bias: bool, optional
    :param device: The device on which the parameters will be placed.
    :type device: torch.device, optional

    :ivar weight: The learnable weights of the module of shape :math:`(\\text{out_features}, \\text{in_features})`.
    :vartype weight: torch.Tensor
    :ivar bias: The learnable bias of the module of shape :math:`(\\text{out_features})`.
        Only present if ``bias`` is ``True``.
    :vartype bias: torch.Tensor
    """
    def __init__(self, in_features, out_features, bias:bool = True, device: torch.device = None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.device = device
        weight = torch.Tensor(out_features, in_features).to(self.device)
        self.weight = nn.Parameter(weight)
        if bias:
            bias = torch.Tensor(out_features).to(self.device)
            self.bias = nn.Parameter(bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        normalized_weight = normalize(self.weight, p=2, dim=1)
        w_times_x= torch.mm(x, normalized_weight.t())
        if self.bias is not None:
            return w_times_x + self.bias
        else:
            return w_times_x

    def __repr__(self):
        device_str = f", device={self.device}" if self.device.type != 'cpu' else ""
        return (f"{self.__class__.__name__}(in_features={self.in_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None}"
                f"{device_str})")
