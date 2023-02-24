import numpy as np
from ..Utils import *

class Loss:
    def __init__(self):
        pass
    def __call__(self, y, y_hat):
        raise NotImplementedError


class NLL(Loss): # should calculate loss on a batch of examples
    def __init__(self):
        super().__init__()
        self.x = None
        self.dx = None
        self.y = None
        self.probs = None
        self.loss = None
    def __call__(self, x, y, mode='test'):
        if(mode == 'train'):
            self.x = x
            self.y = y.reshape(-1)
            exps = np.exp(x - np.max(x, axis = 0, keepdims=True))
            probs = exps / np.sum(exps, axis = 0, keepdims=True)
            log_probs = -np.log(probs)
            #print("Shape of self.y = ", self.y)
            loss = np.sum(log_probs[self.y, np.arange(self.y.shape[0])])
            self.dx = probs
            mask = np.zeros_like(self.dx)
            mask[self.y, np.arange(self.y.shape[0])] = 1
            self.dx = self.dx - mask
            return loss, self.dx
        elif(mode == 'test'):
            self.x = x
            self.y = y.reshape(-1)
            exps = np.exp(x - np.max(x, axis = 0, keepdims=True))
            probs = exps / np.sum(exps, axis = 0, keepdims=True)
            log_probs = -np.log(probs)
            loss = np.sum(log_probs[self.y, np.arange(self.y.shape[0])])
            return loss