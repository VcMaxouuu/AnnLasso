import torch

def function_derivative(func, u):
    u = torch.tensor(u, dtype=torch.float, requires_grad=True)
    y = func(u)
    y.backward()
    return u.grad.item()






