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

class active:
    def __init__(self,X,y, lambda_, sigma_2):
        self.X = X
        self.y = y
        self.lambda_ = lambda_
        self.sigma_2 = sigma_2 
        self.I = np.ones(self.X.shape[1])

        self.Sigma = 0
        self.mu = 0

    def get_posterior(self):
        self.Sigma = np.linalg.inv((self.lambda_ * self.I) + self.sigma_2 * (self.X.T @ self.X))
        self.mu = np.linalg.inv((self.lambda_ * self.sigma_2 * self.I) + (self.X.T @ self.X)) @ (self.X.T @ self.y)



