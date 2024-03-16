import torch 
from torch.nn import MSELoss
import utils
import numpy as np
from torch.nn.functional import normalize, linear

class GeneralModel():
    def __init__(self, name, warm_start, one_ista, learning_rate, n_hidden_units):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.name = name
        self.warm_start = warm_start
        self.one_ista = one_ista

        self.lr = learning_rate
        self.p2 = n_hidden_units
        self.activation = utils.Custom_act_fun()

        self.curves_sd, self.curves_ista = None, None
        self.trained = False
        self.training_time = None
        self.set_name = None


    def predict(self, X):
        X = utils.X_to_tensor(X, device = self.device)
        with torch.inference_mode():
            y_pred = self.forward(X)
        return y_pred

    def forward(self, X):
        layer1_output = self.activation(self.layer1(X))
        w2_normalized = normalize(self.layer2.weight, p=2, dim=1)
        logits = linear(layer1_output, w2_normalized, self.layer2.bias)
        return logits.squeeze().to(X.device)

    def summary(self):
        print("MODEL INFORMATIONS:")
        print('=' * 20)
        print("General:")
        print('―' * 20)
        print(f"  Name: {self.name}")
        print(f"  Training Status: {'Trained' if self.trained else 'Not Trained'}")
        if self.trained:
            print(f"\t Training Time: {self.training_time:.3f} seconds")
            print(f"\t Training Set: {self.set_name if self.set_name is not None else 'N/A'}")
        print(f"  Lambda_qut: {np.round(self.lambda_qut, 4)}\n")
        print("Layers:")
        print('―' * 20)
        if not self.trained:
            print("  Model has not been trained yet. Call 'fit' method first.")
        else:
            print("  Layer 1: ")
            print(f"\t Shape = {list(self.layer1.weight.shape)}")
            imp_feat = utils.important_features(self)
            print(f"\t Number of non zero entries in weights: {imp_feat[0]}")
            print(f"\t Non zero entries indexes: {imp_feat[1]}")
            print("  Layer 2:")
            print(f"\t Shape = {list(self.layer2.weight.shape)}")


    def results_analysis(self, X_test, y_test, true_indexes):
        X_test, y_test = utils.data_to_tensor(X_test, y_test, device=self.device)
        y_hat = self.predict(X_test)
        error_fn = MSELoss(reduction='mean') 
        error = error_fn(y_hat, y_test)

        pred_indexes = utils.important_features(self)[1]

        TP = len(set(pred_indexes) & set(true_indexes))
        FP = len(set(pred_indexes) - set(true_indexes))

        TPR = TP / len(true_indexes) if len(true_indexes) > 0 else 0
        FDR = FP / (TP + FP) if (TP + FP) > 0 else 0
        
        exact_recovery = (set(pred_indexes) == set(true_indexes))

        return {"error": error.item(), "TPR": TPR, "FDR": FDR, "exact_recovery": exact_recovery}

    def fit(self, X_train, y_train, X_test=None, y_test=None, print_epochs=True, set_name=None):
        pass

    def plot_learning_curve(self):
        if self.curves_sd == None and self.curves_ista == None:
            print("No learning curve has been computed yet. Call 'fit' method first.")
            return
        return utils.draw_loss_curves(self.curves_sd, self.curves_ista)

    def apply_feature_selection(self, X):
        pass
    
    def fit_and_apply(self, X, y):
        self.fit(X, y, print_epochs=False, set_name=self.set_name)
        X = self.apply_feature_selection(X)
        return X

    def plot_layer1_status(self):
        return utils.draw_layer1_evolution(self.layer1_history)


    