import numpy as np

class Affine():
    count_layer = 0 # class variable
    def __init__(self, n_in, n_out):
        self.name = 'Affine' + str(Affine.count_layer)
        Affine.count_layer = Affine.count_layer + 1
        self.W = np.random.randn(n_out, n_in) * 0.1
        self.b = np.zeros(n_out).reshape(-1,1)
        self.x = None
        self.dW = None
        self.db = None
        self.out = None

    def forward(self, x):
        self.x = x
        out = np.dot(self.W, x) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(self.W.T, dout)
        self.dW = np.dot(dout, self.x.T)
        self.db = np.sum(dout, axis=1).reshape(-1,1)
        return dx
    
    def get_params(self):
        params = {}
        params[self.name + "_w"] = [self.W, self.dW]
        params[self.name + "_b"] = [self.b, self.db]
        return params   
    def set_params(self, params):
        self.W = params[self.name + "_w"][0]
        self.dW = params[self.name + "_w"][1]
        self.b = params[self.name + "_b"][0]
        self.db = params[self.name + "_b"][1]

    def step(self, lr):
        self.W = self.W - lr * self.dW
        self.b = self.b - lr * self.db
    
    def set_weight_scale(self, scale):
        self.W = np.random.randn(self.W.shape[0], self.W.shape[1]) * scale
        self.b = np.zeros(self.b.shape[0]).reshape(-1,1)
    # @classmethod
    # def get_params(cls):
    #     return cls.__dict__
    
class Sigmoid():
    count_layer = 0
    def __init__(self):
        self.name = 'Sigmoid' + str(Sigmoid.count_layer)
        Sigmoid.count_layer = Sigmoid.count_layer + 1
        self.out = None

    def forward(self, x):
        #unstable version of sigmoid layer
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
    def get_params(self):
        params = {}
        return params
    
    def set_weight_scale(self, scale):
        pass
    
    def step(self, lr):
        pass
    
    def set_params(self, params):
        pass
# class Softmax():
#     def __init__(self, y_labels):
#         self.x = None
#         self.probs = None
#         self.y_labels = y_labels.reshape(-1)
#         self.y_hat = None
#         pass
#     def forward(self, x):
#         #stable version of softmax layer
#         self.x = x
#         exps = np.exp(x - np.max(x, axis = 0, keepdims=True))
#         out = exps / np.sum(exps, axis=0, keepdims=True)
#         self.probs = out    
#         y_hat = np.argmax(out, axis=0)  
#         self.y_hat = y_hat
#         return -np.log(self.probs[self.y_labels, np.arange(self.y_labels.shape[0])])
#     def backward(self,dout):
#         dout = np.ones_like(self.probs)
#         mask = np.zeros_like(self.probs)
#         mask[self.y_labels, np.arange(self.y_labels.shape[0])] = 1
#         dx = self.probs - (mask)
#         return dx