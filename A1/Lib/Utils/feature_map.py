import numpy as np

class biased_feature_map:
    def __init__(self, X) -> None:
        self.X = X
        self.dim = X.shape[1]
        self.num_examples = X.shape[0]
        pass
    def __call__(self):
        return np.concatenate((np.ones((self.num_examples, 1)), self.X), axis = 1)