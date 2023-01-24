import numpy as np
from .. import feature_map as fm

class HingeClassifier:
    def __init__(self, X, y):
        """
        Returns a HingeClassifier object that trains the data using the hinge loss function without any regularization
        `X`: input data
        `y`: labels
        """
        f = fm.biased_feature_map(X)
        self.X = f()
        self.y = y
        self.w = np.zeros(self.X.shape[1])
        self.corr_count  = 0
    def train(self, epochs, lr):
        self.w = np.zeros(self.X.shape[1])
        self.corr_count = 0
        for i in range(epochs):
            flag = False
            for j in range(self.X.shape[0]):
                if(self.y[j]*np.dot(self.w, self.X[j]) < 1):
                    Flag = True
                    self.w += lr*self.y[j]*self.X[j] # softer update step as compared to the perceptron algorithm, # the explanation is in the notebook file
                    self.corr_count = self.corr_count + 1
            if(not Flag):
                break
        return self.w,self.corr_count
    def forward(self, X):
        X_feat = fm.biased_feature_map(X)()
        return (np.dot(X_feat, self.w))