from h1_regression import OLS

from sklearn.datasets import load_boston

data = load_boston()
X = data["data"]
y = data["target"]

rgr = OLS()
rgr.fit(X,y)

print(rgr.theta)