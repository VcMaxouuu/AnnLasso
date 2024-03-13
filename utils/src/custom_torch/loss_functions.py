import torch
import torch.nn as nn

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
            target = torch.tensor(target, dtype=torch.long)
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