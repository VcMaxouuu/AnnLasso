"""
Implements optimization techniques like FISTA
"""

import torch
from torch.optim.optimizer import Optimizer
from .shrinkage_operator import shrinkage_operator


class ISTABlock(Optimizer):
    def __init__(self, params, lambda_, lr, nu, lr_min=1e-9):
        defaults = dict(lambda_=lambda_, nu=nu, lr=lr, lr_min=lr_min)
        super(ISTABlock, self).__init__(params, defaults)

    def step(self, closure):
        """
        The closure should:
        - If called with backward=True, compute the loss and call backward() to accumulate gradients.
        - If called with backward=False, just return the loss without modifying gradients.
        """
        if closure is None:
            raise ValueError("ISTABlock requires a closure for computation.")

        for group in self.param_groups:
            if not group['params']:
                continue

            lambda_ = group['lambda_']
            nu = group['nu']
            lr = group['lr']
            lr_min = group['lr_min']
            params = group['params']

            # Compute loss and gradients at current parameters
            loss, _ = closure(backward=True)

            # Store original parameters and gradients
            original_params = [p.data.clone() for p in params]
            grads = [p.grad.clone() for p in params]

            # Start line search
            current_lr = lr
            while True:
                with torch.no_grad():
                    # Update parameters using the current learning rate
                    for idx, p in enumerate(params):
                        update = original_params[idx] - current_lr * grads[idx]
                        p.data.copy_(shrinkage_operator(update, current_lr * lambda_, nu))

                    # Evaluate the new loss without backward
                    loss_new, _ = closure(backward=False)

                    if loss_new <= loss:
                        # Improvement found or at least non-increasing loss
                        break
                    else:
                        # Reduce learning rate
                        current_lr /= 10.0
                        if current_lr < lr_min:
                            # No improvement found, revert to original params
                            for idx, p in enumerate(params):
                                p.data.copy_(original_params[idx])
                            break

        # Return the loss without backward
        return closure(backward=False)
