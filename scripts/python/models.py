import numpy as np
import cvxpy as cp
from scipy.spatial import distance

class Interpolant():
    
    def __init__(self, kernel, X, Y, jitter=1e-8):
        
        self.kernel = kernel
        self.jitter = jitter
        self.N = X.shape[0]
        self.centers = X
        self.weights = np.linalg.solve(self.kernel(X,X) + self.jitter*np.eye(self.N,self.N),Y)
    
    def __call__(self, x):
        
        fx = np.zeros(x.shape)
        for i in range(self.N):
            fx = fx + np.array(self.weights[i]*self.kernel(self.centers[i].reshape(-1,1),x)).reshape(-1,1)
        return fx
    
    def norm(self):
        
        return np.sqrt(self.weights.T @ self.kernel(self.centers,self.centers) @ self.weights)[0][0]
    
    def _power(self, x):
        
        Kxx = np.diag(self.kernel(x,x))
        KxX = self.kernel(x,self.centers)
        KXx = self.kernel(self.centers,x)
        KXX = self.kernel(self.centers,self.centers) + self.jitter*np.eye(self.centers.shape[0],self.centers.shape[0])
        
        return np.sqrt( Kxx - np.diag( KxX @ np.linalg.solve(KXX,KXx) ))
    
    def bounds(self, x, norm_f):
            
        return (self._power(x) * np.sqrt(norm_f**2 - self.norm()**2)).reshape(-1,1)
    
class Kernel_Ridge_Regressor():
    
    def __init__(self, kernel, X, Y, lam, jitter=1e-8):
        
        self.kernel = kernel
        self.jitter = jitter
        self.N = X.shape[0]
        self.X = X
        self.Y = Y
        self.lam = lam
        
        self.KXX = kernel(X,X) + self.jitter*np.eye(self.N,self.N)
        self.weights = np.linalg.solve(self.kernel(X,X) + self.N*lam*np.eye(self.N,self.N), Y)
    
    def __call__(self, x):
        
        fx = np.zeros(x.shape)
        for i in range(self.N):
            fx = fx + np.array(self.weights[i]*self.kernel(self.X[i].reshape(-1,1),x)).reshape(-1,1)
        return fx
    
    def norm(self):
        
        return np.sqrt(self.weights.T @ self.kernel(self.X,self.X) @ self.weights)[0][0]
    
class Noisy_Bounds():
            
    def __init__(self, X, Y, kernel, gamma, delta_bar):
        
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.K = np.array(kernel(X,X))
        self.gamma = gamma
        self.delta_bar = delta_bar
        
        self.jitter = 1e-6
        self.N = X.shape[0]
        
        nu = cp.Variable((self.N,1))
        obj = cp.Minimize(.25 * cp.quad_form(nu, self.K) + nu.T @ self.Y + self.delta_bar * cp.norm(nu, 1))
        prob = cp.Problem(obj,constraints=[])
        prob.solve() 
        self.Delta = prob.value
        
        self.s = Interpolant(self.kernel, self.X, self.Y, jitter=1e-8)
        
    def _power(self, x):
        
        Kxx = np.diag(self.kernel(x,x))
        KxX = self.kernel(x,self.X)
        KXx = self.kernel(self.X,x)
        KXX = self.kernel(self.X,self.X) + self.jitter*np.eye(self.N,self.N)
        
        return np.sqrt( Kxx - np.diag( KxX @ np.linalg.solve(KXX,KXx) ))
        
    def solve_problem(self, x, model):
        '''For a single x, with shape (1,1)'''
    
        x = x.reshape(1,1)
        return (self._power(x) * np.sqrt(self.gamma**2 + self.Delta)
                + self.delta_bar * np.linalg.norm(np.linalg.solve(self.kernel(self.X,self.X), self.kernel(self.X,x)), ord=1)
                + np.abs(self.s(x) - model(x))
                #TODO add difference between model and interpolant
                ).reshape(-1,1)
        
    def __call__(self, xx, model, dir="upper"):
        
        output = np.empty(xx.shape)
        for i in range(xx.shape[0]):
            output[i,:] = self.solve_problem(xx[i,:], model)
        
        if dir == "upper": return model(xx) + output
        if dir == "lower": return model(xx) - output
    
class Optimal_Bounds():
    
    def __init__(self, X, Y, kernel, gamma, delta_bar):
        
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.gamma = gamma
        self.delta_bar = delta_bar
        
        self.jitter = 1e-6
        self.N = X.shape[0]
        
    def solve_problem(self, x, dir):
        '''For a single x, with shape (1,1)'''
    
        x = x.reshape(1,1)
        dist = distance.cdist(x, self.X).min()
        min_dist = 1e-6
        
        if dist >= min_dist:    
                        
            c = cp.Variable(self.N)
            cx = cp.Variable(1)
            
            Xx = np.append(self.X, x, axis=0)
            K = self.kernel(Xx,Xx)
            K = np.array(K + self.jitter * np.eye(self.N + 1))
            
            if dir == "upper": obj = cp.Maximize(cx.flatten())
            if dir == "lower": obj = cp.Minimize(cx.flatten())
            constraints = [cp.matrix_frac(cp.hstack([c, cx]), K) <= self.gamma**2,
                           cp.norm((c.flatten() - self.Y.flatten()), "inf") <= self.delta_bar]
            
                           
        else:
                        
            c = cp.Variable((self.N,1))
            K = self.kernel(self.X,self.X)
            K = np.array(K + self.jitter * np.eye(self.N))
            
            cost_vec = np.zeros(self.N)
            cost_vec[np.where(distance.cdist(x, self.X) <= min_dist)[1]] = 1
            if dir == "upper": obj = cp.Maximize(cost_vec.T @ c)
            if dir == "lower": obj = cp.Minimize(cost_vec.T @ c)
            constraints = [cp.matrix_frac(c, K) <= self.gamma**2,
                           cp.norm((c.flatten() - self.Y.flatten()), "inf") <= self.delta_bar]
                                       
        prob = cp.Problem(obj,constraints)
        prob.solve()
        return prob.value
    
    def __call__(self, xx, dir="upper"):
        
        output = np.empty(xx.shape)
        for i in range(xx.shape[0]): 
            output[i,:] = self.solve_problem(xx[i,:], dir)
        
        return output