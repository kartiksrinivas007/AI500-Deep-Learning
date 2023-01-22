import numpy as np
import math 

class Dataset:
    def __init__(self, num_points, n_dim=2 ,mode="NL", dist="Normal", p1=None, p2=None, r1=None , r2=None, separability=None,  w_1= None, frac=0.5, bias=None):
        self.p1 = p1
        self.p2 = p2
        self.r1 = r1
        self.r2 = r2
        self.w_1= w_1
        self.mode = mode
        self.dist = dist
        self.frac = frac
        self.n_dim = n_dim
        self.bias = bias
        self.num_points = num_points
        if(self.mode == "NL"):
            if(self.dist == "Normal"):
                self.ds = np.random.multivariate_normal(self.p1, self.p2, self.num_points) 
                f = math.ceil(self.num_points*self.frac)
                self.labels = np.concat(np.ones((f)), -1*np.ones((self.num_points - f)))
            elif(self.dist == "Circular"):
                # create data that is on the border of two circles, add some gaussian noise if needed
                f = math.ceil(num_points* self.frac)
                pts_1 = np.random.randn(f, self.n_dim)
                pts_1 /= np.linalg.norm(pts_1, axis = 1)
                pts_1 *= r1
                pts_2 = np.random.randn(num_points - f, self.n_dim)
                pts_2 /= np.linalg.norm(pts_2, axis = 1)
                pts_2 *= r2
                self.ds = np.concat(pts_1, pts_2, axis = 0) 
                self.labels = np.concat(np.ones((f)), -1*np.ones((self.num_points - f)))
        else:
            # create data that is on both sides of a given line given the actual normal to the hyperplane w_1 provided it passes through the origin
            self.ds = np.random.randn(self.num_points, self.n_dim)
            self.labels = np.sign(np.dot(self.ds, self.w_1))
            #add some separability to the dataset
            #separate positive points to the left of the line and negative points to the right of the line
            self.ds[np.where(self.labels == 1), :] += separability * (self.w_1/np.linalg.norm(self.w_1))
            self.ds[np.where(self.labels == -1), :] -= separability * (self.w_1/np.linalg.norm(self.w_1))
            if(self.bias is not None):
                self.ds += bias*(self.w_1/np.linalg.norm(self.w_1))
    
    def get_data(self) -> tuple:
        return self.ds,self.labels

