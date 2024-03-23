import torch

def function_derivative(func, u, device):
    u = torch.tensor(u, dtype=torch.float, device=device, requires_grad=True)
    y = func(u)
    y.backward()
    return u.grad.item()






