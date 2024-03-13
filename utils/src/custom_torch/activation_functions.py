import torch
import torch.nn as nn

class Custom_act_fun(nn.Module): 
    """Custom torch activation function.
    
    Parameters
    ----------
    M (int, optional): Parameter for the softplus function. Defaults to 20.
    k (int, optional): Power to which the softplus function is raised. Defaults to 1.
    u_0 (int, optional): Softplus shift constant. Defaults to 1.
    
    """
    def __init__(self, M=20, k=1, u_0=1):
        super(Custom_act_fun, self).__init__() 
        self.M = M
        self.k = torch.tensor(k, dtype=torch.float)
        self.u_0 = torch.tensor(u_0, dtype=torch.float)
        self.softplus = nn.Softplus(beta=self.M)  
  
    def forward(self, u: torch.Tensor): 
        """Forward pass of the custom activation function."""
        self.k.to(u.device)
        self.u_0.to(u.device)
        return (1/self.k) * (self.softplus(u + self.u_0).pow(self.k) - self.softplus(self.u_0 * torch.ones_like(u)).pow(self.k))
