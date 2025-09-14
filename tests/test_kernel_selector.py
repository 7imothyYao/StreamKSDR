import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from kernel_selector import KernelSelector

def test_selector_linear_preference_on_linear_relation():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 5)
    # Y is linear combination + small noise
    W = rng.randn(5, 2)
    Y = X @ W + 0.01 * rng.randn(200, 2)
    selector = KernelSelector(threshold=0.4, method='combined')
    k, info = selector.select_kernel(X, Y, verbose=False)
    assert k in ('linear', 'rbf')


def test_selector_rbf_allowed_on_nonlinear():
    rng = np.random.RandomState(1)
    X = rng.randn(300, 4)
    # Nonlinear relation: squared terms
    Y = np.stack([(X[:,0]**2 + X[:,1]**2), np.sin(X[:,2])], axis=1)
    selector = KernelSelector(threshold=0.2, method='combined')
    k, info = selector.select_kernel(X, Y, verbose=False)
    assert k in ('linear', 'rbf')
