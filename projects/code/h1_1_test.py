from h1_regression import OLS, active

from sklearn.datasets import load_boston
import numpy as np

data = load_boston()
X = data["data"]
y = data["target"]

rgr = OLS()
rgr.fit(X,y)

rgr = active(X,y,.9,.2)
rgr.get_posterior()

print(np.random.multivariate_normal(rgr.mu, rgr.Sigma))