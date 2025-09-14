"""Minimal kernel selector (pure NumPy, no tuning search).

Provides a single heuristic: compute a linearity score (mix of max abs corr
and a lightweight single-dimension R^2) then threshold to choose 'linear' or 'rbf'.
All advanced scoring, confidence gymnastics, and class wrapper removed to keep
core minimal and dependency‑free.
"""

import numpy as np


def _linearity_score(X, Y):
    # Max absolute correlation part
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    Xs = np.sqrt((Xc**2).sum(axis=0, keepdims=True) / (X.shape[0]-1) + 1e-8)
    Ys = np.sqrt((Yc**2).sum(axis=0, keepdims=True) / (Y.shape[0]-1) + 1e-8)
    Xn = Xc / Xs
    Yn = Yc / Ys
    corr = (Xn.T @ Yn) / X.shape[0]
    max_corr = np.max(np.abs(corr), axis=0)
    mcorr = max_corr.mean()
    # Simple one-dim R^2 using first Y (fallback if degenerate)
    y = Y[:,0] if Y.ndim==2 and Y.shape[1]>0 else Y.ravel()
    X1 = np.column_stack([X, np.ones(X.shape[0])])
    try:
        XtX = X1.T @ X1
        beta = np.linalg.inv(XtX + 1e-8*np.eye(XtX.shape[0])) @ X1.T @ y
        pred = X1 @ beta
        ss_tot = ((y - y.mean())**2).sum()
        ss_res = ((y - pred)**2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-8)
        r2 = max(0.0, min(1.0, r2))
    except np.linalg.LinAlgError:
        r2 = 0.0
    return 0.7 * mcorr + 0.3 * r2


def auto_select_kernel_numpy(X, Y, threshold=0.4, verbose=False):
    s = _linearity_score(X, Y)
    k = 'linear' if s >= threshold else 'rbf'
    if verbose:
        print(f"Kernel={k} (score={s:.3f} threshold={threshold:.3f})")
    return k, {'score': s, 'kernel': k, 'threshold': threshold}


def select_kernel_for_data(X, Y, threshold=0.4):
    return 'linear' if _linearity_score(X, Y) >= threshold else 'rbf'


class KernelSelector:
    def __init__(self, threshold=0.4):
        self.threshold = threshold
    def select_kernel(self, X, Y, verbose=False):
        return auto_select_kernel_numpy(X, Y, self.threshold, verbose)
    def check_linearity(self, X, Y):
        return _linearity_score(X, Y)


if __name__ == "__main__":
    # Demonstration code
    print("Minimal kernel selector demo (no typing dependency)")

    # Generate test data
    np.random.seed(42)

    # Strongly linear data
    X_linear = np.random.randn(1000, 10)
    Y_linear = X_linear[:, :3] @ np.array([[1.5], [2.0], [1.2]]) + 0.1 * np.random.randn(1000, 1)
    Y_linear = np.hstack([Y_linear, Y_linear * 0.8 + 0.1 * np.random.randn(1000, 2)])

    # Nonlinear data
    X_nonlinear = np.random.randn(1000, 10)
    Y_nonlinear = np.zeros((1000, 3))
    Y_nonlinear[:, 0] = np.sin(X_nonlinear[:, 0]) * np.cos(X_nonlinear[:, 1])
    Y_nonlinear[:, 1] = np.exp(-X_nonlinear[:, 2]**2 / 2)
    Y_nonlinear[:, 2] = np.tanh(X_nonlinear[:, 3] * X_nonlinear[:, 4])

    # Test selector
    selector = KernelSelector()

    print("\nStrongly linear data:")
    kernel1, info1 = selector.select_kernel(X_linear, Y_linear, verbose=True)

    print("\nNonlinear data:")
    kernel2, info2 = selector.select_kernel(X_nonlinear, Y_nonlinear, verbose=True)

    # Minimal interface test
    print("\nMinimal interface test:")
    print(f"Strongly linear data kernel: {select_kernel_for_data(X_linear, Y_linear)}")
    print(f"Nonlinear data kernel: {select_kernel_for_data(X_nonlinear, Y_nonlinear)}")
