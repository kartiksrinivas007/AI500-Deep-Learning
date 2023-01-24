import numpy as np
from ..feature_map import  * # you can change the feature_map later if you want
class Perceptron:
    def __init__(self, X, y) -> None:
        f = biased_feature_map(X)
        self.X = f()
        self.y = y
        self.dim = self.X.shape[1]
        self.num_examples = self.X.shape[0]
        self.weights = np.zeros(self.dim)
        self.corr_count = 0
        pass
    def train(self, max_iter = 1000) -> np.ndarray:
        self.weights = np.zeros(self.dim)
        self.corr_count = 0
        for i in range(max_iter):
            flag = False
            for j in range(self.X.shape[0]):
                if(self.y[j]*np.dot(self.X[j], self.weights) <= 0):
                    flag = True
                    self.weights += self.y[j]*self.X[j]
                    self.corr_count = self.corr_count + 1
            if(flag == False):
                break
        return self.weights,self.corr_count
    
    def forward(self, X) -> np.ndarray:
        f = biased_feature_map(X)
        X = f()
        return (np.dot(X, self.weights))