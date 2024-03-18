import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import linear, normalize
import utils
from torch.autograd import grad


class RegressionAnn(nn.Module):
    def __init__(
        self, X, y, lambda_, learning_rate=0.01, Sd=True, n_hidden_units=20, 
        init_weights=None, iniscale=0.01, rel_err=1.e-12, n_ista=5000
        ):
        super().__init__()
        self.device = X.device

        # Parameters
        self.lamb = lambda_
        self.p2 = n_hidden_units
        self.lr = learning_rate
        self.Sd = Sd
        self.iniscale = iniscale
        self.rel_err = rel_err
        self.n_Ista = n_ista
        self.X, self.y = X, y
        self.n_features, self.p1 = X.shape

        # Pytorch Parameters
        self.activation = utils.Custom_act_fun()
        if init_weights is not None:
            self.layer1, self.layer2 = init_weights
        else:
            self.layer1 = nn.Linear(self.p1, self.p2, device = self.device, dtype = torch.float64)
            self.layer2 = nn.Linear(self.p2, 1, device = self.device, dtype = torch.float64)

    
    def fit(self, print_epochs=True):
        loss_fn = utils.CustomRegressionLoss(lambda_=self.lamb).to(self.device)
        bare_loss_fn = nn.MSELoss(reduction = 'mean').to(self.device)

        #### Gradient descent part
        cost_fun_history, train_loss_history, epochs_history, layer1_history = [],[],[],{'weight': [], 'bias': []}
        epoch, best_loss = 0, float('inf')
        min_lr = 0.0001
        optimizer = torch.optim.SGD(params=self.parameters(), lr=self.lr, momentum=0.9, dampening=0, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.99, min_lr=min_lr)
        while self.Sd and epoch<=1500: #! Maximum epoch set for testing purposes:
            y_pred = self(self.X)
            loss = loss_fn(y_pred, self.y, self.layer1)
            bare_loss = bare_loss_fn(y_pred, self.y)
            optimizer.zero_grad()
            loss.backward()

            if epoch % 100 == 0:
                cost_fun_history.append(loss.item())
                train_loss_history.append(bare_loss.item())
                epochs_history.append(epoch)
                layer1_history['weight'].append(self.layer1.weight.data.clone().detach().cpu().numpy())
                layer1_history['bias'].append(self.layer1.bias.data.clone().detach().cpu().numpy())
                
                if epoch != 0:
                    if print_epochs:
                        print(f"\tEpoch: {epoch} | Loss: {loss:.5f} | MSE on train set : {bare_loss:.5f} | learning rate : {optimizer.param_groups[0]['lr']:.6f}")
                    if optimizer.param_groups[0]['lr'] == min_lr:
                        if loss < best_loss:
                            best_loss = loss
                        else:
                            if print_epochs:
                                print("\n\tGradient descent stopped: minimum learning rate reached and loss is no longer decreasing.\n")
                            break

            if loss < 1e-5 :
                if print_epochs:
                    print("\n\tGradient descent stopped: loss is zero.\n")
                break

            epoch +=1
            optimizer.step()
            scheduler.step(loss) 

        curves_sd = {'epochs': np.array(epochs_history), 'cost': np.array(cost_fun_history), 'train': np.array(train_loss_history)}
        
        #### FISTA part
        cost_fun_history, train_loss_history, epochs_history = [],[],[]
        if self.n_Ista>0:
            lr, epoch, best_loss = min_lr, 0, float('inf')
            optimizer_penalized = utils.FISTA(params=self.layer1.parameters(), lambda_=self.lamb, lr=lr)
            optimizer_unpenalized = utils.FISTA(params=self.layer2.parameters(), lambda_=0.0, lr=lr)
            while True:
                y_pred = self(self.X)
                loss = loss_fn(y_pred, self.y, self.layer1)
                bare_loss = bare_loss_fn(y_pred, self.y)

                gradients_bare_loss = grad(bare_loss, self.layer1.parameters(), retain_graph=True)
                gradients_loss = grad(loss, self.layer2.parameters(), retain_graph=True)

                for param, grad_value in zip(self.layer1.parameters(), gradients_bare_loss):
                    param.grad = grad_value
                for param, grad_value in zip(self.layer2.parameters(), gradients_loss):
                    param.grad = grad_value

                if epoch % 100 == 0:
                    cost_fun_history.append(loss.item())
                    train_loss_history.append(bare_loss.item())
                    epochs_history.append(epoch)
                    layer1_history['weight'].append(self.layer1.weight.data.clone().detach().cpu().numpy())
                    layer1_history['bias'].append(self.layer1.bias.data.clone().detach().cpu().numpy())

                    if epoch != 0:
                        if print_epochs:
                            print(f"\tEpoch FISTA: {epoch} | Loss: {loss:.5f} | MSE on train set : {bare_loss:.5f} | w1 non zeros entries : {utils.important_features(self)} | learning rate : {optimizer_penalized.param_groups[0]['lr']:.6f}")
                        if loss > best_loss:
                            optimizer_penalized.param_groups[0]['lr'] *= .9
                            optimizer_unpenalized.param_groups[0]['lr'] *= .9
                        if torch.abs(loss-best_loss)/loss < self.rel_err:
                            if print_epochs:
                                print('\n\tISTA stopped: relative error reached.')
                            break

                        best_loss = loss

                if epoch == self.n_Ista:
                    if print_epochs:
                        print("\n\tISTA stopped : maximum ISTA iterations reached.")
                    break
                
                optimizer_penalized.step()
                optimizer_unpenalized.step()
                epoch += 1

        curves_ista = {'epochs': np.array(epochs_history), 'cost':np.array(cost_fun_history),'train': np.array(train_loss_history)}

        return (curves_sd, curves_ista, layer1_history)

              
    def forward(self, X):
        layer1_output = self.activation(self.layer1(X))
        w2_normalized = normalize(self.layer2.weight, p=2, dim=1)
        logits = linear(layer1_output, w2_normalized, self.layer2.bias)
        return logits.squeeze().to(X.device)