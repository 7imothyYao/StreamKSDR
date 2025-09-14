import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mainFunction.OKS_main import OnlineKernelSDR

def test_online_shape_linear():
    X = np.random.randn(40, 8)
    Y = np.random.randn(40, 3)
    model = OnlineKernelSDR(d_x=8, d_y=3, k=5,
                            D_x=8, D_y=3,  # Linear kernel requires D_x=d_x, D_y=d_y
                            sigma_x=1.0, sigma_y=1.0,
                            kernel_x='linear', kernel_y='linear',
                            base_lr=0.01, adaptive_lr=False,
                            random_state=0)
    for i in range(len(X)):
        model.update(X[i], Y[i])
    # Manual projection like in ds.py: Phi_x @ U
    Phi = model.rff_x.transform(X)
    Z = (Phi - model.mean_x) @ model.U
    assert Z.shape == (40, 5)

def test_online_does_not_crash_rbf():
    X = np.random.randn(30, 6)
    Y = np.random.randn(30, 2)
    model = OnlineKernelSDR(d_x=6, d_y=2, k=4,
                            D_x=48, D_y=24,
                            sigma_x=1.0, sigma_y=1.0,
                            kernel_x='rbf', kernel_y='rbf',
                            base_lr=0.01, adaptive_lr=False,
                            random_state=42)
    for i in range(len(X)):
        model.update(X[i], Y[i])
    # Manual projection: Phi_x @ U
    Phi = model.rff_x.transform(X)
    Z = (Phi - model.mean_x) @ model.U
    assert Z.shape == (30, 4)
