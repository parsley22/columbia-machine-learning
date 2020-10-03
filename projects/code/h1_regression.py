import numpy as np

class OLS:
    def __init__(self):
        self.theta = 0

    def fit(self, X, Y, lamdba_ = 0):
        N = X.shape[0]
        p = X.shape[1]
        self.theta = np.linalg.inv((lamdba_ * np.eye(p)) + X.T @ X) @ X.T @ Y

    def predict(self, X):
        return X.dot(self.theta)
