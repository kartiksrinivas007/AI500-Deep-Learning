from ..Utils import *
import numpy as np

class SGD:
    def __init__(self, model, lr, mode='normal', momentum=None):
        self.lr = lr
        self.mode = mode
        self.momentum = momentum
        self.v = None
        self.t = 0
        self.params = model.params # not holding a reference earlier! , no it will hold a perfect reference of what model.params is pointing to
        #of no significance however since modle.params does not hold a reference to layer.params
        self.model= model # should hold a reference to the real object made
    
    def step(self):
        i = 0
        for layer in self.model.layers: # layer is a  copy of a model layer object(which is real and the one being used for the forward pass)
            layer_params = layer.get_params() # get the values of the copy of the original layer
            params = {}
            for key,value in layer_params.items():
                params[key] = [value[0] - self.lr * value[1], np.zeros_like(value[1])]
                # resetting the gradients
            layer.set_params(params) # set the values to the copy 
            self.model.set_layer(layer, i) # you replace the layer in the model with the copied layer
            i = i + 1