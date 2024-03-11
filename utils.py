import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.style.use('https://github.com/dhaitz/matplotlib-stylesheets/raw/master/pacoty.mplstyle')
plt.rcParams['figure.dpi'] = 300 
plt.rcParams['axes.titlesize'] = 20  
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['figure.figsize'] = (12, 8)


def data_to_tensor(X, y=None, device=None):
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

    if y is not None:
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).type(torch.float64)
        elif isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            y = torch.from_numpy(y.values.squeeze()).type(torch.float64)
        else:
            y = torch.as_tensor(y, dtype=torch.float64)

        if device:
            y = y.to(device)
    else:
        y = None
    return X, y

class Custom_act_fun(nn.Module): 
    def __init__(self, M=20, k=1, u_0=1): 
        super(Custom_act_fun, self).__init__() 
        self.M = M
        self.k = torch.tensor(k, dtype=torch.float)
        self.u_0 = torch.tensor(u_0, dtype=torch.float)
        self.softplus = nn.Softplus(beta=self.M)  
  
    def forward(self, u): 
        self.k.to(u.device)
        self.u_0.to(u.device)
        return (1/self.k) * (self.softplus(u + self.u_0).pow(self.k) - self.softplus(self.u_0 * torch.ones_like(u)).pow(self.k))


def function_derivative(func, u):
    u = torch.tensor(u, dtype=torch.float, requires_grad=True)
    y = func(u)
    y.backward()
    return u.grad.item()


class CustomClassificationLoss(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lamb = lambda_
        self.entropy = nn.CrossEntropyLoss(reduction = 'sum')

    def forward(self, input, target, w1):
        if target.dtype != torch.long:
            target = torch.tensor(target, dtype=torch.long)
        cross_entropy_loss = self.entropy(input, target)
        lasso_regularization = self.lamb * (torch.abs(w1.weight).sum() + torch.abs(w1.bias).sum())
        total_loss = cross_entropy_loss + lasso_regularization
        return total_loss

class CustomRegressionLoss(nn.Module):
    def __init__(self, lambda_):
        super().__init__()
        self.lamb = lambda_
        self.mse_loss = nn.MSELoss(reduction='sum') 

    def forward(self, input, target, w1):
        """
        Args:
            input (torch.Tensor): _description_
            target (torch.Tensor): _description_
        """
        square_root_lasso_loss = torch.sqrt(self.mse_loss(input, target))
        lasso_regularization = self.lamb * (torch.abs(w1.weight).sum() + torch.abs(w1.bias).sum())
        total_loss = square_root_lasso_loss + lasso_regularization
        return total_loss

class FISTA(torch.optim.Optimizer):
    def __init__(self, params, lr, lambda_):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if lambda_ < 0.0:
            raise ValueError(f"Invalid lambda: {lambda_} - should be >= 0.0")
        
        defaults = dict(lr=lr, lambda_=lambda_)
        super(FISTA, self).__init__(params, defaults)
    
    def shrinkage_operator(self, u, tresh):
        return torch.sign(u) * torch.maximum(torch.abs(u) - tresh, torch.tensor(0.0, device=u.device))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                lr = group['lr']
                lambda_ = group['lambda_']
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state = self.state[p]
                    state['x_prev'] = p.data.clone()
                    state['y_prev'] = p.data.clone()
                    state['t_prev'] = torch.tensor(1., device=p.device)

                x_prev, y_prev, t_prev = state['x_prev'], state['y_prev'], state['t_prev']

                x_next = self.shrinkage_operator(y_prev - lr * grad, lr*lambda_)
                t_next = (1. + torch.sqrt(1. + 4. * t_prev ** 2)) / 2.
                y_next = x_next + ((t_prev - 1) / t_next) * (x_next - x_prev)

                state['x_prev'], state['y_prev'], state['t_prev'] = x_next, y_next, t_next

                p.data.copy_(x_next)

        return loss

def lambda_qut_regression(X, act_fun=Custom_act_fun(), n_samples=100000, mini_batch_size=500, alpha=0.05, option='quantile'):
    """
    Args:
        X (torch.Tensor): 
    """
    offset = 0 if n_samples % mini_batch_size == 0 else 1
    n_samples_per_batch = n_samples // mini_batch_size + offset
    
    n, p1 = X.shape    
    #### Array to store the computed statistics for each generated sample.
    fullList = torch.zeros((mini_batch_size*n_samples_per_batch,))

    #### Monte Carlo Simulation
    for index in range(n_samples_per_batch):
        y_sample = torch.normal(mean=0., std=1, size=(n, 1, mini_batch_size)).type(torch.float64)
        y = (torch.mean(y_sample, dim=0)- y_sample).to(X.device)
        xy = y * X.unsqueeze(2).expand(-1, -1, mini_batch_size)
        xy_max = torch.amax(torch.abs(torch.sum(xy, dim=0)), dim=0)
        norms = torch.norm(y, p=2, dim=0)
        fullList[index * mini_batch_size:(index+1) * mini_batch_size]= xy_max/norms
    
    fullList = fullList * function_derivative(act_fun, 0)
 
    if option=='full':
        return fullList
    elif option=='quantile':
        return np.quantile(fullList, 1-alpha)
    else:
        pass

def lambda_qut_classification():
    pass


def generate_linear_data(save_dir=None, n_train=100, n_val=30, n_test=300, p1=200, SNR=3):
    if save_dir is None:
        raise ValueError("No file directory has been given")
    if not isinstance(save_dir, str):
        raise TypeError("File directory must be a string")


    for s in range(1, 18):
        beta = SNR * np.ones(s)
        inds = np.random.choice(range(p1), s, replace=False)
        datasets = {} 

        for n, name in zip([n_train, n_val, n_test], ['train', 'val', 'test']):
            x = np.random.normal(size=(n, p1))
            y = np.dot(x[:, inds], beta) + np.random.normal(size=(n))
            datasets[name] = [pd.DataFrame(x), pd.DataFrame(y)]

        s_dir = os.path.join(save_dir, f"s{s}")
        os.makedirs(s_dir, exist_ok=True)
        
        for name in ['train', 'val', 'test']:
            x, y = datasets[name]
            x.to_csv(os.path.join(s_dir, f"{s}-{name}-x.csv"), header=False, index=False)
            y.to_csv(os.path.join(s_dir, f"{s}-{name}-y.csv"), header=False, index=False)

        inds_df = pd.DataFrame(inds)
        inds_df.to_csv(os.path.join(s_dir, f"{s}-important_inds.csv"), index=False)

        print(f"Data for s={s} saved in directory: {s_dir}")


def predict(model, X):
    np.set_printoptions(suppress=True, precision=4) 
    X, _ = data_to_tensor(X, device = model.device)
    with torch.inference_mode():
        y_pred = model.forward(X)
    return y_pred

def important_features(model):
    weight = model.layer1.weight.data
    non_zero_columns = torch.any(weight != 0, dim=0)
    indices = torch.where(non_zero_columns)[0]
    count = torch.sum(non_zero_columns).item()
    return count, sorted(indices.tolist())


def draw_curvers(curves_sd, curves_ista, lambda_):
    epochs_sd = curves_sd['epochs']
    train_loss_sd = curves_sd['train']
    test_loss_sd = curves_sd['test']
    
    epochs_ista = curves_ista['epochs']
    train_loss_ista = curves_ista['train']
    test_loss_ista = curves_ista['test']

    if len(epochs_sd) != 0:
        epochs_ista_adjusted = np.array(epochs_ista) + epochs_sd[-1]
    else:
        epochs_ista_adjusted = epochs_ista

    plt.figure(figsize=(10, 6))

    if len(epochs_sd) != 0:
        plt.plot(epochs_sd, train_loss_sd, color='#5A5B9F', linewidth=2, label='Train Loss SD')
    plt.plot(epochs_ista_adjusted, train_loss_ista, color='#D94F70', linewidth=2, label='Train Loss ISTA')

    if len(test_loss_sd)!=0:
        plt.plot(epochs_sd, test_loss_sd, color='#5A5B9F', linestyle='dashed', linewidth=2, label='Test Loss SD')
        plt.plot(epochs_ista_adjusted, test_loss_ista, color='#D94F70', linestyle="dashed", linewidth=2, label='Test Loss ISTA')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Loss Curves for SD and ISTA Phases. Lambda = {np.round(lambda_, 4)}')
    plt.legend()
    plt.show()