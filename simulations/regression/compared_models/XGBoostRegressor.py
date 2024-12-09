from xgboost import XGBRegressor as XGBR
from sklearn.preprocessing import StandardScaler
from numpy import nonzero, full
from boruta import BorutaPy

class XGBoostRegressor():
    def __init__(self):
        self.model = XGBR()
        self.boruta_selector = None
        self.mean_y = None
        self.no_features_selected = False

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        xgb = XGBR()
        self.boruta_selector = BorutaPy(xgb, n_estimators=100, verbose=0)
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
            return full(X.shape[0], self.mean_y)
        else:
            X = self.scaler.transform(X)
            X_selected = self.boruta_selector.transform(X)
            return self.model.predict(X_selected)

    def _important_features(self):
        selected_features = nonzero(self.boruta_selector.support_)[0]
        return len(selected_features), list(selected_features)
