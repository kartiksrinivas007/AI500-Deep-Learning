import numpy as np
from ...Utils import *
from ...Solvers import *
from ...Gates import *

class FFNN:
    def __init__(self, layers):
        """
        The model musthave a certain architecture, which should be specified by the user, in terms of the classes that are needed
        to build the model, the types of objects need to be specified to the model as input, so an array must be passed as input,
        so that the appropriate layers are made
        it should be called like this:
        nn = FFNN([
            Affine(2, 3),
            Sigmoid(),
            Affine(3,1),
            softmax()
        ])
        nn.get_params() should return a list of all the parameters of the model
        nn.get_params() should contain pairs of parameters and their gradients
        nn.forward(x) should return the output of the model
        nn.backward(dout) should return the gradients of the model
        nn.get_grads() should return a list of all the gradients of the model
        """
        self.layers  = layers
        pass
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    def get_params(self):
        
        pass
    def backward(self, dout):
        pass
        # for layer in reversed(self.layers):
        #     dout = layer.backward(dout)
        # return dout