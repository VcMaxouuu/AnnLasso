import torch 
import torch.nn as nn

class FISTA(torch.optim.Optimizer):
    """FISTA optimizer. This optimizer implements the Fast Iterative Shrinkage-Thresholding Algorithm.

    Parameters
    ----------
    params (iterable) : Iterable of parameters to optimize or dicts defining parameter groups.
    lr (float) : Learning rate.
    lambda_ (float) : Regularization strength.

    """
    def __init__(self, params, lr, lambda_):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if lambda_ < 0.0:
            raise ValueError(f"Invalid lambda: {lambda_} - should be >= 0.0")
        
        defaults = dict(lr=lr, lambda_=lambda_)
        super(FISTA, self).__init__(params, defaults)
    
    def shrinkage_operator(self, u, tresh):
        """Shrinkage operator for the FISTA algorithm.

        Args:
            u (torch.Tensor): Input tensor.
            tresh (float): Treshold value.

        Returns:
            torch.Tensor: Output tensor.
        """
        return torch.sign(u) * torch.maximum(torch.abs(u) - tresh, torch.tensor(0.0, device=u.device))

    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional)

        Returns:
            float : Loss value.
        """
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