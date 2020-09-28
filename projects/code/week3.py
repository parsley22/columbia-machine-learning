"""
This file impliments a ridge regression model from scratch
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

X = load_boston().data 
y = load_boston().target

X_train, X_test, y_train, y_test = train_test_split(X,y)

class OLS:
    def __init__(self, X,Y):
        self.loss = "rss"
        self.X = X
        self.Y = Y

        p = self.X.shape[1]
        N = self.X.shape[0]
        self.theta = np.random.randn(p)

    def fit(self):
        self.theta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.Y

    def predict(self, X):
        return X.dot(self.theta)

def evaluate(theta, X_test, y_test):

    rse_cumsum = []
    for i in range(X_test.shape[0]):
        x = X_test[i,:]
        y = y_test[i]

        y_hat = theta.dot(x)
        rse = (y - y_hat)**2
        rse_cumsum.append(rse)
    print(sum(rse_cumsum)/X_test.shape[0])

rgr = OLS(X_train, y_train)
rgr.fit()

evaluate(rgr.theta, X_test, y_test)