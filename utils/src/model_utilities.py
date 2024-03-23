import torch
import torch.nn as nn
from torch.nn.functional import linear, normalize
from torch.autograd import grad
import utils
import numpy as np

def important_features(layer1):
    """Get the indices of the most important features for a given model.

    Args:
        layer1 (torch.nn.Linear): First layer of a model.

    Returns:
        int: Number of important features.
        list: List of important feature indices.

    """
    weight = layer1.weight.data
    non_zero_columns = torch.any(weight != 0, dim=0)
    indices = torch.where(non_zero_columns)[0]
    count = torch.sum(non_zero_columns).item()
    return count, sorted(indices.tolist())


def train_model(mode, X, y, lambda_, lr=0.01, p2=20, Sd=True, n_ista=-1,
    print_epochs=False, init_weights=None, rel_err=1.e-12):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, y = X.to(device), y.to(device)

    n_features, p1 = X.shape
    activation = utils.Custom_act_fun(device)

    if mode == "classification":
        loss_fn = utils.CustomClassificationLoss(lambda_=lambda_).to(device)
        bare_loss_fn = nn.CrossEntropyLoss(reduction = 'mean').to(device)
    else:
        loss_fn = utils.CustomRegressionLoss(lambda_=lambda_).to(device)
        bare_loss_fn = nn.MSELoss(reduction = 'mean').to(device)
        

    if init_weights is not None:
        layer1, layer2 = init_weights
        layer1, layer2 = layer1.to(device), layer2.to(device)
    else:
        layer1 = nn.Linear(p1, p2, device=device, dtype=torch.float64)
        layer2 = nn.Linear(p2, 1, device=device, dtype=torch.float64)


    def forward(X, layer1, layer2):
        layer1_output = activation(layer1(X))
        w2_normalized = normalize(layer2.weight, p=2, dim=1)
        logits = linear(layer1_output, w2_normalized, layer2.bias)
        return logits.squeeze()


    #### Fitting - Gradient descent ####
    min_lr = 0.0001
    cost_fun_history, train_loss_history, epochs_history, layer1_history = [],[],[],{'weight': [], 'bias': []}
    

    if Sd:
        epoch, best_loss = 0, float('inf')
        optimizer = torch.optim.SGD(params=[*layer1.parameters(), *layer2.parameters()], lr=lr, momentum=0.9, dampening=0, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4, factor=0.99, min_lr=min_lr)

        while True and epoch<=1500: #! Maximum epoch set for testing purposes:
            optimizer.zero_grad()

            y_pred = forward(X, layer1, layer2)
            loss = loss_fn(y_pred, y, layer1)
            bare_loss = bare_loss_fn(y_pred, y)

            loss.backward()
            optimizer.step()
            scheduler.step(loss) 

            if epoch % 100 == 0:
                cost_fun_history.append(loss.item())
                train_loss_history.append(bare_loss.item())
                epochs_history.append(epoch)
                layer1_history['weight'].append(layer1.weight.data.clone().detach().cpu().numpy())
                layer1_history['bias'].append(layer1.bias.data.clone().detach().cpu().numpy())

                if epoch != 0:
                    if print_epochs:
                        print(f"\tEpoch: {epoch} | Loss: {loss:.5f} | MSE on train set : {bare_loss:.5f} | learning rate : {optimizer.param_groups[0]['lr']:.6f}")
                    if optimizer.param_groups[0]['lr'] == min_lr:
                        if loss < best_loss: # First stopping criterion; loss not decreasing and minimum learning rate reached.
                                best_loss = loss
                        else:
                            if print_epochs:
                                print("\n\tGradient descent stopped: minimum learning rate reached and loss is no longer decreasing.\n")
                            break
            
            if loss < 1e-5 : # Second stopping criterion; loss is "zero".
                if print_epochs:
                    print("\n\tGradient descent stopped: loss is zero.\n")
                break
            
            
            epoch += 1

    curves_sd = {'epochs': np.array(epochs_history), 'cost': np.array(cost_fun_history), 'train': np.array(train_loss_history)}

    #### Fitting - FISTA ####
    cost_fun_history, train_loss_history, epochs_history = [],[],[]
    if n_ista > 0:
        epoch, best_loss = 0, float('inf')
        optimizer_penalized = utils.FISTA(params=layer1.parameters(), lambda_=lambda_, lr=min_lr)
        optimizer_unpenalized = utils.FISTA(params=layer2.parameters(), lambda_=0.0, lr=min_lr)
        while True:
            y_pred = forward(X, layer1, layer2)
            loss = loss_fn(y_pred, y, layer1)
            bare_loss = bare_loss_fn(y_pred, y)

            if epoch % 100 == 0:
                cost_fun_history.append(loss.item())
                train_loss_history.append(bare_loss.item())
                epochs_history.append(epoch)
                layer1_history['weight'].append(layer1.weight.data.clone().detach().cpu().numpy())
                layer1_history['bias'].append(layer1.bias.data.clone().detach().cpu().numpy())

                if epoch != 0:
                    if print_epochs:
                        print(f"\tEpoch FISTA: {epoch} | Loss: {loss:.5f} | MSE on train set : {bare_loss:.5f} | w1 non zeros entries : {utils.important_features(layer1)} | learning rate : {optimizer_penalized.param_groups[0]['lr']:.6f}")
                    if loss > best_loss:
                        optimizer_penalized.param_groups[0]['lr'] *= .9
                        optimizer_unpenalized.param_groups[0]['lr'] *= .9
                    if torch.abs(loss-best_loss)/loss < rel_err:
                        if print_epochs:
                            print('\n\tISTA stopped: relative error reached.')
                        break

                    best_loss = loss

            if epoch == n_ista:
                if print_epochs:
                    print("\n\tISTA stopped : maximum ISTA iterations reached.")
                break

            gradients_bare_loss = grad(bare_loss, layer1.parameters(), retain_graph=True)
            gradients_loss = grad(loss, layer2.parameters(), retain_graph=True)

            for param, grad_value in zip(layer1.parameters(), gradients_bare_loss):
                param.grad = grad_value
            for param, grad_value in zip(layer2.parameters(), gradients_loss):
                param.grad = grad_value
            
            optimizer_penalized.step()
            optimizer_unpenalized.step()
            epoch += 1

    curves_ista = {'epochs': np.array(epochs_history), 'cost':np.array(cost_fun_history),'train': np.array(train_loss_history)}

    return {"model": (layer1, layer2), "curves": (curves_sd, curves_ista, layer1_history)}
