from Utils import *
import numpy as np

class Affine:
    def __init__(self, n_in, n_out):
        self.W = np.random.randn(n_in, n_out) * 0.01
        self.b = np.zeros(n_out)
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return dx
    
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        #unstable version of sigmoid layer
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx