# OKS_batch module for Online Kernel Supervised PCA
# This module implements the Online Kernel Supervised PCA algorithm using Random Fourier Features.
import numpy as np
from .rff import RFF
    
class BatchKSDR:
    def __init__(self,
                 d_x: int, # input dimension
                 d_y: int, # output dimension
                 k: int, # number of components
                 D_x: int, # input feature dimension
                 D_y: int, # output feature dimension
                 sigma_x: float = 1.0, # input feature bandwidth
                 sigma_y: float = 1.0, # output feature bandwidth
                 kernel_x: str = 'rbf', # input kernel type
                 kernel_y: str = 'rbf', # output kernel type
                 random_state=12):
        seed_x = random_state
        seed_y = random_state + 1
        
        self.rff_x = RFF(d_x, D_x, sigma_x, kernel_x, seed_x)
        self.rff_y = RFF(d_y, D_y, sigma_y, kernel_y, seed_y)
        self.k = k
        
        self.U = None
        self.mean_x = None
        self.mean_y = None
        
    def fit(self, X: np.ndarray, Y: np.ndarray) -> 'BatchKSDR':
        n_samples = X.shape[0]
        
        Phi = self.rff_x.transform(X)
        Psi = self.rff_y.transform(Y)
        
        self.mean_x = np.mean(Phi, axis=0)
        self.mean_y = np.mean(Psi, axis=0)
        
        Phi_c = Phi - self.mean_x
        Psi_c = Psi - self.mean_y
        
        C_xy = (Phi_c.T @ Psi_c) / n_samples
        self.C_xy = C_xy
        # 使用更稳定的计算方式
        M = C_xy @ C_xy.T
        
        # 导入 eigh
        from scipy.linalg import eigh
        
        eigenvals, eigenvecs = eigh(M)
        idx = np.argsort(eigenvals)[::-1]
        self.U = eigenvecs[:, idx[:self.k]]

        return self