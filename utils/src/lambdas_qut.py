import torch
from .custom_torch import Custom_act_fun
from .others import function_derivative

def lambda_qut_regression(X, n_samples=100000, mini_batch_size=500, alpha=0.05, option='quantile'):
    """Computes the quantile universal treshold value for regression.

    Args:
        X (torch.Tensor): Input data
        act_fun (torch.nn.Module, optional): Activation function. Defaults to Custom_act_fun().
        n_samples (int, optional): Number of samples to use for Monte Carlo simulation. Defaults to 100000.
        mini_batch_size (int, optional): Size of mini-batches for Monte Carlo simulation. Defaults to 500.
        alpha (float, optional): Quantile value. Defaults to 0.05.
        option (str, optional): Whether to return the full list of statistics or just the quantile. Defaults to 'quantile'.

    """
    device = X.device
    act_fun=Custom_act_fun(device)
    offset = 0 if n_samples % mini_batch_size == 0 else 1
    n_samples_per_batch = n_samples // mini_batch_size + offset
    
    n, p1 = X.shape    
    fullList = torch.zeros((mini_batch_size*n_samples_per_batch,), device=device)

    for index in range(n_samples_per_batch):
        y_sample = torch.normal(mean=0., std=1, size=(n, 1, mini_batch_size)).type(torch.float64)
        y = (torch.mean(y_sample, dim=0)- y_sample).to(device)
        xy = y * X.unsqueeze(2).expand(-1, -1, mini_batch_size)
        xy_max = torch.amax(torch.abs(torch.sum(xy, dim=0)), dim=0)
        norms = torch.norm(y, p=2, dim=0)
        fullList[index * mini_batch_size:(index+1) * mini_batch_size]= xy_max/norms
    
    fullList = fullList * function_derivative(act_fun, 0, device)
 
    if option=='full':
        return fullList
    elif option=='quantile':
        return torch.quantile(fullList, 1-alpha)
    else:
        pass

def lambda_qut_classification():
    pass