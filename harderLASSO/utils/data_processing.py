"""
Contains utilities for transforming data into tensors and preparing inputs for models.
"""

import numpy as np
import torch
import pandas as pd
from typing import Optional, Union

class StandardScaler:
    """
    A standard scaler for normalizing tensor data by removing the mean and scaling to unit variance.

    :param mean: The mean value(s) to be used for scaling. Calculated during fit if not provided. Default is ``None``.
    :type mean: Optional[torch.Tensor]
    :param std: The standard deviation value(s) to be used for scaling. Calculated during fit if not provided. Default is ``None``.
    :type std: Optional[torch.Tensor]
    :param epsilon: A small constant added to the standard deviation to prevent division by zero. Default is ``1e-7``.
    :type epsilon: float
    """
    def __init__(self, mean: Optional[torch.Tensor] = None, std: Optional[torch.Tensor] = None, epsilon: float = 1e-7):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values: torch.Tensor) -> None:
        """
        Computes and stores the mean and standard deviation of the provided values.

        :param values: The input tensor to calculate the mean and standard deviation.
        :type values: torch.Tensor
        """
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims, correction=0)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        """
        Transforms the values by scaling them based on the computed mean and standard deviation.

        :param values: The input tensor to transform.
        :type values: torch.Tensor
        :return: The transformed tensor with mean zero and unit variance.
        :rtype: torch.Tensor
        """
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values: torch.Tensor) -> torch.Tensor:
        """
        Fits the scaler on the provided values and then transforms them.

        :param values: The input tensor to fit and transform.
        :type values: torch.Tensor
        :return: The transformed tensor with mean zero and unit variance.
        :rtype: torch.Tensor
        """
        self.fit(values)
        return self.transform(values)


def X_to_tensor(X: Union[pd.DataFrame, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Converts input data X to a PyTorch tensor of dtype float.

    :param X: The input data in DataFrame, ndarray, or tensor format.
    :type X: Union[pd.DataFrame, np.ndarray, torch.Tensor]
    :return: The input data converted to a float tensor.
    :rtype: torch.Tensor
    """
    if isinstance(X, pd.DataFrame):
        X = torch.tensor(X.values, dtype= torch.float)
    elif isinstance(X, torch.Tensor):
        X = X.float()
    else:
        X = torch.tensor(X, dtype=torch.float)
    return X


def target_to_tensor(target: Union[pd.Series, pd.DataFrame, list, np.ndarray, torch.Tensor]) -> torch.Tensor:
    """
    Converts input target data to a PyTorch tensor of dtype float.

    :param target: The target data in Series, DataFrame, list, ndarray, or tensor format.
    :type target: Union[pd.Series, pd.DataFrame, list, np.ndarray, torch.Tensor]
    :return: The target data converted to a float tensor.
    :rtype: torch.Tensor
    """
    if isinstance(target, (pd.Series, pd.DataFrame)):
        target = target.values.squeeze()
    if isinstance(target, (list, np.ndarray)):
        target = np.array(target).flatten()

    return torch.tensor(target, dtype=torch.float)


def get_hat_p(target: Union[torch.Tensor, np.ndarray, pd.Series]) -> torch.Tensor:
    """
    Computes the class distribution vector `hat_p` for classification tasks.

    The target should have classes from 0 to `num_classes` without empty classes.

    :param target: The target labels in tensor, ndarray, or Series format.
    :type target: Union[torch.Tensor, np.ndarray, pd.Series]
    :return: A tensor representing the relative frequency of each class.
    :rtype: torch.Tensor
    """
    if isinstance(target, torch.Tensor):
        target = target.to(torch.int64)
    else:
        target = torch.tensor(target, dtype=torch.int64)

    n_items = len(target)
    n_classes = len(target.unique())
    class_counts = torch.zeros(n_classes, dtype=torch.float)

    for class_index in range(n_classes):
        class_counts[class_index] = (target == class_index).sum()

    hat_p = class_counts / n_items
    return hat_p
