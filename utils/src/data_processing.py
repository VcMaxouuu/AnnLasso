import numpy as np
import torch

def X_to_tensor(X, device=None):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X).type(torch.float64)
    elif isinstance(X, pd.DataFrame):
        X = torch.from_numpy(X.values).type(torch.float64)
    else:
        X = torch.as_tensor(X, dtype=torch.float64)
    if device:
        X = X.to(device)
    norms = torch.norm(X, p=2, dim=0)
    X = X / norms
    return X

def y_to_tensor(y, device=None):
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).type(torch.float64)
    elif isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
        y = torch.from_numpy(y.values.squeeze()).type(torch.float64)
    else:
        y = torch.as_tensor(y, dtype=torch.float64)
    if device:
        y = y.to(device)
    return y


def data_to_tensor(X, y=None, device=None):
    X = X_to_tensor(X, device)
    y = y_to_tensor(y, device)
    return X, y