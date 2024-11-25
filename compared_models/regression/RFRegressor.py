from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.preprocessing import StandardScaler
import numpy as np
from boruta import BorutaPy

class RandomForestRegressor():
    def __init__(self):
        self.model = RFR()
        self.boruta_selector = None
        self.mean_y = None
        self.no_features_selected = False

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        rf = RFR()
        self.boruta_selector = BorutaPy(rf, n_estimators=100, verbose=0)
        self.boruta_selector.fit(X, y)

        X_selected = self.boruta_selector.transform(X)

        # Check if any features were selected
        if X_selected.shape[1] == 0:
            self.mean_y = y.mean()
            self.no_features_selected = True
            self.imp_feat = (0, [])
        else:
            self.no_features_selected = False
            self.model.fit(X_selected, y)
            self.imp_feat = self._important_features()

    def predict(self, X):
        if self.no_features_selected:
            return np.full(X.shape[0], self.mean_y)
        else:
            X = self.scaler.transform(X)
            X_selected = self.boruta_selector.transform(X)
            return self.model.predict(X_selected)

    def _important_features(self):
        selected_features = np.nonzero(self.boruta_selector.support_)[0]
        return len(selected_features), list(selected_features)
