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
    def __init__(self,X,y):
        self.X = X
        self.y = y
        
        self.I = np.ones(self.X.shape[1])
        self.posterior = None

    def get_posterior(self, lambda_, sigma2):
        Sigma = np.linalg.inv((lambda_ * self.I) + sigma2 * (self.X.T @ self.X))
        mu = np.linalg.inv((lambda_ * sigma2 * self.I) + (self.X.T @ self.X)) @ (self.X.T @ self.y)
        self.posterior = [mu,Sigma]
        
    def best_predict(self, D, lambda_, sigma2):
        best_x_val = 0
        best_x_ix = 0
        
        for i,x in enumerate(D):
            sigma2_0 = sigma2 + (x.T @ self.posterior[1] @ x)
            if sigma2_0 > best_x_val:
                best_x_val = sigma2_0
                best_x_ix = i
                
        return best_x_val, best_x_ix
    
    def predict_y(self,x, sigma2):
      
        sigma2_0 = sigma2 + (x.T @ (self.posterior[1])@ x)
        mu_0 = x.T.dot(self.posterior[0])
        
        y_0 = np.random.normal(mu_0, sigma2_0)
        return y_0
    
    def update_posterior(self, x_0, y_0, lambda_, sigma2):
        
        Sigma = np.linalg.inv((lambda_*self.I) + 1/sigma2 * (x_0.dot(x_0.T) + (self.X.T.dot(self.X))))
        mu = np.linalg.inv((lambda_ * sigma2 * self.I) + (x_0.dot(x_0.T) + self.X.T.dot(self.X))) * (x_0.dot(y_0) + self.X.T.dot(self.y))
        
        self.posterior = [mu, Sigma]
        
    def fit(self, D, lambda_, sigma2):
        locations = []
        
        if self.posterior is None:
                self.get_posterior(lambda_, sigma2)
        
        for i in range(10):
            x_0,loc = self.best_predict(D, lambda_, sigma2)
            y_0 = self.predict_y(D[loc,:], sigma2)
            self.update_posterior(D[loc,:],y_0, lambda_, sigma2)
            locations.append(loc)
            
        return locations
                
        
                
                



