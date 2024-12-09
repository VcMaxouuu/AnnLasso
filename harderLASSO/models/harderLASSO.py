from .base import RegressionModel, ClassifierModel, CoxModel

class harderLASSORegressor(RegressionModel):
    """Model with non-linear penalty used for regression"""
    def __init__(self, hidden_dims=(20,), nu=0.1, lambda_qut=None):
        super().__init__(hidden_dims=hidden_dims, nu=nu, lambda_qut=lambda_qut)

class harderLASSOClassifier(ClassifierModel):
    """Model with non-linear penalty used for classification"""
    def __init__(self, hidden_dims=(20,), nu=0.1, lambda_qut=None):
        super().__init__(hidden_dims=hidden_dims, nu=nu, lambda_qut=lambda_qut)

class harderLASSOCOX(CoxModel):
    """Model with non-linear penalty used for survival analysis"""
    def __init__(self, hidden_dims=(20,), nu=0.1, lambda_qut=None):
        super().__init__(hidden_dims=hidden_dims, nu=nu, lambda_qut=lambda_qut)
