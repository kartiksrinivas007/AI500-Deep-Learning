import numpy as np
np.random.seed(0)

class BatchSampler:
    def __init__(self, X, y) -> None:
        self.X = X
        self.y = y.reshape(1,-1)
        self.mask = None
        pass
    
    def __call__(self, batch_size):
        # returns a list of batches required for one total epoch after shuffling
        self.mask = np.random.permutation(self.X.shape[1])
        self.X = self.X[:,self.mask]
        self.y = self.y[:,self.mask]
        batches = []
        for i in range(0, self.X.shape[1], batch_size):
            if( i + batch_size > self.X.shape[1] ):
                batches.append( (self.X[:,i:], self.y[:,i:]) )
            else:
                batches.append((self.X[:,i:i+batch_size], self.y[:,i:i+batch_size]))
        return batches