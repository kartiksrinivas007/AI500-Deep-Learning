import numpy as np
from ...Utils import *
from ...Solvers import *
from ...Gates import *
from collections import ChainMap

class FFNN:
    def __init__(self, layers, weight_scale=0.1):
        """
        The model musthave a certain architecture, which should be specified by the user, in terms of the classes that are needed
        to build the model, the types of objects need to be specified to the model as input, so an array must be passed as input,
        so that the appropriate layers are made
        it should be called like this:
        ```
        nn = FFNN([
            Affine(2, 3),
            Sigmoid(),
            Affine(3,1),
            softmax()
        ])
        ```
        nn.get_params() should return a list of all the parameters of the model
        nn.get_params() should contain pairs of parameters and their gradients
        nn.forward(x) should return the output of the model
        nn.backward(dout) should return the gradients of the model
        nn.get_grads() should return a list of all the gradients of the model
        """
        self.layers  = layers
        self.weight_scale = weight_scale
        self.params = {}
    
    def forward(self, x):
        i = 0
        for layer in self.layers:
            x = layer.forward(x) # does the forward and updates the parameters of the proxy layer
            self.set_layer(layer, i) #set the actual layer to the proxy layer
            i = i + 1
        return x

    def get_params(self): # returns a copy fo the state of the Neural Network, not the original
        for layer in self.layers:
            layer_params = layer.get_params()
            self.params = {**self.params, **layer_params}
        return self.params
    
    def __str__(self):
        string = ""
        for layer in self.layers:
            string += str(layer.name) +" "
        string += "\n"
        for layer in self.layers:
            layer_params = layer.get_params()
            for param,value in layer_params.items():
                string += "param: " + str(param) + " value: " + str(value[0].shape) + str(value) + "\n"
        return string
    def backward(self, dout):
        i = len(self.layers) - 1
        for layer in reversed(self.layers):
            dout  = layer.backward(dout) # modifies the proxy layer
            self.set_layer(layer, i) # sets the actual layer to the proxy layer
            i =  i - 1
        return dout
    
    def set_layer(self,layer, index):
        self.layers[index] = layer
    
    # def reset_grads(self): #meaningless since they are not references to the actual layer elements present inside the layers
    #     for key, value in self.params.items():
    #         self.params[key][1] = np.zeros_like(value[0])
