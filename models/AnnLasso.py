import torch
from torch.nn import Linear
import numpy as np
import utils
import time
from tqdm import tqdm
from .sub_models import GeneralModel, RegressionAnn

class AnnLassoRegression(GeneralModel):
    def __init__(self, name=None, warm_start=True, one_ista=True, learning_rate=0.01, n_hidden_units=20):
        super().__init__(name, warm_start, one_ista, learning_rate, n_hidden_units)
       

    def fit(self, X_train, y_train, print_epochs=True, set_name=None):
        self.set_name = set_name

        #### Data treatement ####
        X_train, y_train = utils.data_to_tensor(X_train, y_train, self.device)

        #### Lambda qut ####
        self.lambda_qut = utils.lambda_qut_regression(X_train)

        start_time = time.time()
        #### Training process ####
        if self.warm_start:
            n_ista = 5000 if not self.one_ista else -1
            iterable = tqdm(range(-1, 6), desc="Lambda path progress") if not print_epochs else range(-1, 6)
            self.curves_sd  = {np.exp(i)/(1+np.exp(i))*self.lambda_qut: None for i in range(-1, 6)}
            self.curves_ista = {np.exp(i)/(1+np.exp(i))*self.lambda_qut: None for i in range(-1,6)}
            self.layer1_history = {np.exp(i)/(1+np.exp(i))*self.lambda_qut: None for i in range(-1,6)}
            
            for i in iterable:
                lambi=np.exp(i)/(1+np.exp(i))*self.lambda_qut
                if print_epochs:
                    print(f"Lambda = {np.round(lambi, 4)} :")
                if i==-1:
                    model = RegressionAnn(X_train, y_train, lambi, self.lr, n_hidden_units=self.p2, n_ista=n_ista)
                elif i==5:
                    model = RegressionAnn(X_train, y_train, self.lambda_qut, self.lr, Sd=False, n_hidden_units=self.p2, n_ista=10000, init_weights=[model.layer1, model.layer2])
                else:
                    model = RegressionAnn(X_train, y_train, lambi, self.lr, n_hidden_units=self.p2, n_ista=n_ista, init_weights=[model.layer1, model.layer2])
                curves_sd, curves_ista, layer1_history = model.fit(print_epochs)
                self.curves_sd[lambi] = curves_sd
                self.curves_ista[lambi] = curves_ista
                self.layer1_history[lambi] = layer1_history
        else:
            model = RegressionAnn(X_train, y_train, self.lambda_qut, self.lr, n_hidden_units=self.p2, n_ista=10000)
            curves_sd, curves_ista, layer1_history = model.fit(print_epochs)
            self.curves_sd = curves_sd
            self.curves_ista = curves_ista
            self.layer1_history = layer1_history

        #### Model parameters attribution ####
        self.layer1 = Linear(model.layer1.in_features, model.layer1.out_features, device=self.device)
        self.layer2 = Linear(model.layer2.in_features, model.layer2.out_features, device=self.device)

        self.layer1.weight.data, self.layer1.bias.data = model.layer1.weight.data.clone(), model.layer1.bias.data.clone()
        self.layer2.weight.data, self.layer2.bias.data = model.layer2.weight.data.clone(), model.layer2.bias.data.clone()

        self.training_time = time.time() - start_time

        self.trained = True
        print("MODEL FITTED !")


class AnnLassoClassification(GeneralModel):
    pass