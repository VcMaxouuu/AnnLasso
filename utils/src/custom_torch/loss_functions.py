import torch
import torch.nn as nn

def custom_penalty_fun(theta, nu=0.1):
    nu = torch.tensor(nu, dtype=torch.float, device=theta.device)
    abs_theta = torch.sum(torch.abs(theta))
    return abs_theta/(1+abs_theta**(1-nu))


class CustomClassificationLoss(nn.Module):
    """Custom loss function for classification tasks. Combines cross-entropy loss with L1 regularization.

    Parameters
    ----------
    lambda_ (float) : Regularization strength.

    """
    def __init__(self, lambda_):
        super().__init__()
        self.lamb = lambda_
        self.entropy = nn.CrossEntropyLoss(reduction = 'sum')

    def forward(self, input, target, layer1):
        """Forward pass of the custom loss function.

        Args:
            input (torch.Tensor) : Output of the neural network.
            target (torch.Tensor) : Ground truth labels.
            layer1 (torch.nn.Linear) : First layer of the neural network.

        Returns:
            torch.Tensor : Total loss.

        """
        if target.dtype != torch.long:
            target = target.long()
        cross_entropy_loss = self.entropy(input, target)
        lasso_regularization = self.lamb * (torch.abs(layer1.weight).sum() + torch.abs(layer1.bias).sum())
        total_loss = cross_entropy_loss + lasso_regularization
        return total_loss

class CustomRegressionLoss(nn.Module):
    """Custom loss function for regression tasks. Combines MSE loss with L1 regularization.

    Parameters
    ----------
    lambda_ (float) : Regularization strength.

    """
    def __init__(self, lambda_):
        super().__init__()
        self.lamb = lambda_
        self.mse_loss = nn.MSELoss(reduction='sum') 

    def forward(self, input, target, layer1):
        """Forward pass of the custom loss function.

        Args:
            input (torch.Tensor) : Output of the neural network.
            target (torch.Tensor) : Ground truth labels.
            layer1 (torch.nn.Linear) : First layer of the neural network.

        Returns:
            torch.Tensor : Total loss.

        """
        square_root_lasso_loss = torch.sqrt(self.mse_loss(input, target))
        lasso_regularization = self.lamb * (torch.abs(layer1.weight).sum() + torch.abs(layer1.bias).sum())
        total_loss = square_root_lasso_loss + lasso_regularization
        return total_loss


class CustomRegressionLoss2(nn.Module):
    """Custom loss function for regression tasks. Combines MSE loss with our custom regularization.

    Parameters
    ----------
    lambda_ (float) : Regularization strength.

    """
    def __init__(self, lambda_):
        super().__init__()
        self.lamb = lambda_
        self.mse_loss = nn.MSELoss(reduction='sum') 

    def forward(self, input, target, layer1):
        """Forward pass of the custom loss function.

        Args:
            input (torch.Tensor) : Output of the neural network.
            target (torch.Tensor) : Ground truth labels.
            layer1 (torch.nn.Linear) : First layer of the neural network.

        Returns:
            torch.Tensor : Total loss.

        """
        square_root_lasso_loss = torch.sqrt(self.mse_loss(input, target))
        regularization = self.lamb * (custom_penalty_fun(layer1.weight) + custom_penalty_fun(layer1.bias))
        total_loss = square_root_lasso_loss + regularization
        return total_loss