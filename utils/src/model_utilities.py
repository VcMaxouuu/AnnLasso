import torch
import numpy as np

def important_features(model):
    """Get the indices of the most important features for a given model.

    Args:
        model (torch.nn.Module): Trained model.

    Returns:
        int: Number of important features.
        list: List of important feature indices.

    """
    weight = model.layer1.weight.data
    non_zero_columns = torch.any(weight != 0, dim=0)
    indices = torch.where(non_zero_columns)[0]
    count = torch.sum(non_zero_columns).item()
    return count, sorted(indices.tolist())