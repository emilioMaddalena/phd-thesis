import numpy as np
from scipy.spatial.distance import cdist

class Kernel_se():
    
    def __init__(self):
        
        self.name = "isotropic squared exponential"
        self.length_scale = np.array([0.5])
        
    def __call__(self, x1, x2):
        '''
        Calculates the squared Mahalanobis distance between x1 and x2.
        In other words, the square of the Euclidean distance where each
        dimension is weighted by l.
        - x1, x2 are expected to have shape (d, nx)
        where nx is the x dimension and d is the num of data-points 
        '''
        
        if x1.ndim == 1 : x1 = x1.reshape(-1,1)
        if x2.ndim == 1 : x2 = x2.reshape(-1,1)
        
        dists = cdist(x1 / self.length_scale, x2 / self.length_scale, metric="sqeuclidean")
        K = np.exp(-0.5 * dists)
        return K
    
    def estimate_hyperparameters(self, dataset, hyperparameter_range):
        pass
        