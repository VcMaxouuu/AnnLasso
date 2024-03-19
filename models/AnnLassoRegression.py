import torch
from torch.nn import Linear
import numpy as np
import utils
import time
from tqdm import tqdm
from models.ModelArchitecture import ModelArchitecture

class AnnLassoRegression(ModelArchitecture):
    def __init__(self, name=None, warm_start=True, one_ista=True, lr=0.01, p2=20):
        super().__init__(name, warm_start, one_ista, lr, p2)
       

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
                    d = utils.train_model("regression", X_train, y_train, lambi, self.lr, self.p2, True, n_ista, print_epochs)                    
                elif i==5:
                    d = utils.train_model("regression", X_train, y_train, self.lambda_qut, self.lr, self.p2, False, 10000, print_epochs, init_weights=[layer1, layer2])   
                else:
                    d = utils.train_model("regression", X_train, y_train, lambi, self.lr, self.p2, True, n_ista, print_epochs, init_weights=[layer1, layer2])   
                layer1, layer2 = d["model"]
                curves_sd, curves_ista, layer1_history = d["curves"]
                self.curves_sd[lambi] = curves_sd
                self.curves_ista[lambi] = curves_ista
                self.layer1_history[lambi] = layer1_history
        else:
            d = utils.train_model("regression", X_train, y_train, self.lambda_qut, self.lr, self.p2, True, 10000, print_epochs)   
            layer1, layer2 = d["model"]
            curves_sd, curves_ista, layer1_history = d["curves"]
            self.curves_sd = curves_sd
            self.curves_ista = curves_ista
            self.layer1_history = layer1_history

        #### Model parameters attribution ####
        self.layer1 = Linear(layer1.in_features, layer1.out_features, device=self.device)
        self.layer2 = Linear(layer2.in_features, layer2.out_features, device=self.device)

        self.layer1.weight.data, self.layer1.bias.data = layer1.weight.data.clone(), layer1.bias.data.clone()
        self.layer2.weight.data, self.layer2.bias.data = layer2.weight.data.clone(), layer2.bias.data.clone()

        self.training_time = time.time() - start_time

        self.trained = True
        self.imp_feat = utils.important_features(self.layer1)
        self.layer1_simplified = Linear(self.imp_feat[0], self.p2, device=self.device)
        self.layer1_simplified.weight.data, self.layer1_simplified.bias.data = self.layer1.weight.data[:, self.imp_feat[1]].clone(), self.layer1.bias.data.clone()
        
        print("MODEL FITTED !")
