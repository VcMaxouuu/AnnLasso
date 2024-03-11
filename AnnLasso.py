import torch
import torch.nn as nn
import numpy as np
import utils
import time
from tqdm import tqdm

class AnnLasso():
    def __init__(self, name=None, mode='regression', warm_start=True, learning_rate=0.01, n_hidden_units=20):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.name = name
        self.mode = mode
        self.warm_start = warm_start

        self.lr = learning_rate
        self.p2 = n_hidden_units

        self.trained = False
        self.training_time = None
       

    def fit(self, X_train, y_train, X_test=None, y_test=None, print_epochs=True, set_name=None, graph=False):
        X_train, y_train = utils.data_to_tensor(X_train, y_train, self.device)
        if X_test is not None:
            X_test, y_test = utils.data_to_tensor(X_test, y_test, self.device)

        self.set_name = set_name
        
        if self.mode=="regression":
            import RegressionModel as rm
            start_time = time.time()
            self.lambda_qut = utils.lambda_qut_regression(X_train)

            if self.warm_start:
                if not print_epochs:
                    iterable = tqdm(range(-1, 6), desc="Lambda path progress")
                else:
                    iterable = range(-1, 6)
                for i in iterable:
                    lambi=np.exp(i)/(1+np.exp(i))*self.lambda_qut
                    if print_epochs:
                        print(f"Lambda = {np.round(lambi, 4)} :")
                    if i==-1:
                        model = rm.AnnLassoRegressionModel(X_train, y_train, lambi, X_test, y_test, self.lr, n_hidden_units=self.p2)
                    elif i==5:
                        model = rm.AnnLassoRegressionModel(X_train, y_train, self.lambda_qut, X_test, y_test, self.lr, Sd=False, n_hidden_units=self.p2, n_ista=10000, init_weights=[model.layer1, model.layer2])
                    else:
                        model = rm.AnnLassoRegressionModel(X_train, y_train, lambi, X_test, y_test, self.lr, n_hidden_units=self.p2, init_weights=[model.layer1, model.layer2])
                    model.fit(print_epochs, graph)
            else:
                model = rm.AnnLassoRegressionModel(X_train, y_train, self.lambda_qut, X_test, y_test)
                model.fit(print_epochs, graph)

            self.layer1 = nn.Linear(model.layer1.in_features, model.layer1.out_features, device=self.device)
            self.layer2 = nn.Linear(model.layer2.in_features, model.layer2.out_features, device=self.device)

            self.layer1.weight.data, self.layer1.bias.data = model.layer1.weight.data.clone(), model.layer1.bias.data.clone()
            self.layer2.weight.data, self.layer2.bias.data = model.layer2.weight.data.clone(), model.layer2.bias.data.clone()

            self.training_time = time.time() - start_time
            self.trained = True

            self.child_model = model
            print("MODEL FITTED !")

        else:
            print("Classification model not implemented yet...")
            pass

    
    def predict(self, X):
        return utils.predict(self.child_model, X)

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
        error_fn = nn.MSELoss(reduction='mean') 
        error = error_fn(y_hat, y_test)

        pred_indexes = utils.important_features(self)[1]

        TP = len(set(pred_indexes) & set(true_indexes))
        FP = len(set(pred_indexes) - set(true_indexes))

        TPR = TP / len(true_indexes) if len(true_indexes) > 0 else 0
        FDR = FP / (TP + FP) if (TP + FP) > 0 else 0
        
        exact_recovery = (set(pred_indexes) == set(true_indexes))

        return {"error": error.item(), "TPR": TPR, "FDR": FDR, "exact_recovery": exact_recovery}