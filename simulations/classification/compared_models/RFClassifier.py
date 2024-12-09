from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.preprocessing import StandardScaler
import numpy as np
from boruta import BorutaPy

class RandomForestClassifier():
    def __init__(self):
        self.model = RFC()
        self.boruta_selector = None
        self.most_frequent_class = None
        self.no_features_selected = False

    def fit(self, X, y):
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(X)

        rf = RFC()
        self.boruta_selector = BorutaPy(rf, n_estimators=100, verbose=0)
        self.boruta_selector.fit(X, y)

        X_selected = self.boruta_selector.transform(X)

        # Check if any features were selected
        if X_selected.shape[1] == 0:
            # Calculate the most frequent class in y
            self.most_frequent_class = np.argmax(np.bincount(y))
            self.no_features_selected = True
            self.imp_feat = (0, [])
        else:
            self.no_features_selected = False
            self.model.fit(X_selected, y)
            self.imp_feat = self._important_features()

    def predict(self, X):
        if self.no_features_selected:
            # Return the most frequent class
            return np.full(X.shape[0], self.most_frequent_class)
        else:
            X = self.scaler.transform(X)
            X_selected = self.boruta_selector.transform(X)
            return self.model.predict(X_selected)

    def _important_features(self):
        selected_features = np.nonzero(self.boruta_selector.support_)[0]
        return len(selected_features), list(selected_features)
