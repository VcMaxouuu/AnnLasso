import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import linear, normalize
import utils
from torch.autograd import grad


class RegressionAnn(nn.Module):
    def __init__(
        self, X, y, lambda_, X_test = None, y_test= None, learning_rate=0.01, Sd=True, n_hidden_units=20, 
        init_weights=None, iniscale=0.01, rel_err=1.e-9, n_ista=5000
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

        if X_test is not None and y_test is not None: 
            self.X_test, self.y_test = X_test, y_test

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
        train_loss_history, test_loss_history, epochs_history = [],[],[]
        epoch, best_loss = 0, float('inf')
        min_lr = 0.0001
        optimizer = torch.optim.SGD(params=self.parameters(), lr=self.lr, momentum=0.9, dampening=0, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.99, min_lr=min_lr)
        while self.Sd and epoch< 15000: ### ! Epoch limited for testing purposes. Need to be removed later.
            y_pred = self(self.X)
            loss = loss_fn(y_pred, self.y, self.layer1)
            bare_loss = bare_loss_fn(y_pred, self.y)
            optimizer.zero_grad()
            loss.backward()

            test_loss = bare_loss_fn(self(self.X_test), self.y_test) if hasattr(self, 'X_test') else ''

            if epoch % 100 == 0:
                train_loss_history.append(bare_loss.item())
                epochs_history.append(epoch)

                if hasattr(self, 'X_test'):
                    test_loss_history.append(test_loss.item())
                
                if epoch != 0:
                    if print_epochs:
                        if test_loss == '':
                            print(f"\tEpoch: {epoch} | Loss: {loss:.5f} | MSE on train set : {bare_loss:.5f} | learning rate : {optimizer.param_groups[0]['lr']:.6f}")
                        else:
                            print(f"\tEpoch: {epoch} | Loss: {loss:.5f} | MSE on train set : {bare_loss:.5f} | MSE on test set : {test_loss:.5f} | learning rate : {optimizer.param_groups[0]['lr']:.6f}")
                    if optimizer.param_groups[0]['lr'] == min_lr:
                        if loss < best_loss:
                            best_loss = loss
                        else:
                            if print_epochs:
                                print("\n\tGradient descent stopped: loss is no longer decreasing.\nMoving to ISTA.\n\n")
                            break

            if loss < 1e-5 :
                if print_epochs:
                    print("\n\tGradient descent stopped: loss is zero.\nMoving to ISTA.\n\n")
                break

            epoch +=1
            optimizer.step()
            scheduler.step(loss) 

        curves_sd = {'epochs': np.array(epochs_history), 'train': np.array(train_loss_history), 'test': np.array(test_loss_history)}
        
        #### FISTA part
        train_loss_history, test_loss_history, epochs_history = [],[],[]
        if self.n_Ista>0:
            lr, epoch, best_loss = min_lr, 0, float('inf')
            optimizer_penalized = utils.FISTA(params=self.layer1.parameters(), lambda_=self.lamb, lr=lr)
            optimizer_unpenalized = utils.FISTA(params=self.layer2.parameters(), lambda_=0.0, lr=lr)
            while True:
                y_pred = self(self.X)
                loss = loss_fn(y_pred, self.y, self.layer1)
                bare_loss = bare_loss_fn(y_pred, self.y)

                test_loss = bare_loss_fn(self(self.X_test), self.y_test) if hasattr(self, 'X_test') else ''

                gradients_bare_loss = grad(bare_loss, self.layer1.parameters(), retain_graph=True)
                gradients_loss = grad(loss, self.layer2.parameters(), retain_graph=True)

                for param, grad_value in zip(self.layer1.parameters(), gradients_bare_loss):
                    param.grad = grad_value
                for param, grad_value in zip(self.layer2.parameters(), gradients_loss):
                    param.grad = grad_value

                if epoch % 100 == 0:
                    train_loss_history.append(bare_loss.item())
                    epochs_history.append(epoch)

                    if hasattr(self, 'X_test'):
                        test_loss_history.append(test_loss.item())

                    if epoch != 0:
                        if print_epochs:
                            if test_loss == '':
                                print(f"\tEpoch FISTA: {epoch} | Loss: {loss:.5f} | MSE on train set : {bare_loss:.5f} | w1 non zeros entries : {utils.important_features(self)} | learning rate : {optimizer_penalized.param_groups[0]['lr']:.6f}")
                            else:
                                print(f"\tEpoch FISTA: {epoch} | Loss: {loss:.5f} | MSE on train set : {bare_loss:.5f} | MSE on test set : {test_loss:.5f} | w1 non zeros entries : {utils.important_features(self)} | learning rate : {optimizer_penalized.param_groups[0]['lr']:.6f}")
                        if loss > best_loss:
                            optimizer_penalized.param_groups[0]['lr'] *= .99
                            optimizer_unpenalized.param_groups[0]['lr'] *= .99
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

        curves_ista = {'epochs': np.array(epochs_history), 'train': np.array(train_loss_history), 'test': np.array(test_loss_history)}

        return (curves_sd, curves_ista)

              
    def forward(self, X):
        layer1_output = self.activation(self.layer1(X))
        w2_normalized = normalize(self.layer2.weight, p=2, dim=1)
        logits = linear(layer1_output, w2_normalized, self.layer2.bias)
        return logits.squeeze().to(X.device)