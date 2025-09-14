"""
Kernel Selector (renamed from kernel_selector_minimal)
Lightweight, NumPy-only automatic kernel selection utilities.
"""
import numpy as np

# --- Core scoring ---

def compute_linearity_score_numpy(X, Y, method='combined'):
    if method == 'max_corr':
        Xc = X - X.mean(axis=0, keepdims=True)
        Yc = Y - Y.mean(axis=0, keepdims=True)
        X_std = np.sqrt(np.sum(Xc**2, axis=0, keepdims=True) / (X.shape[0]-1) + 1e-8)
        Y_std = np.sqrt(np.sum(Yc**2, axis=0, keepdims=True) / (Y.shape[0]-1) + 1e-8)
        Xn = Xc / X_std
        Yn = Yc / Y_std
        corr = (Xn.T @ Yn) / X.shape[0]
        score = float(np.mean(np.max(np.abs(corr), axis=0)))
    elif method == 'explained_var':
        y = Y[:,0] if (Y.ndim==2 and Y.shape[1]>0) else Y.ravel()
        X1 = np.column_stack([X, np.ones(X.shape[0])])
        try:
            XtX = X1.T @ X1
            beta = np.linalg.inv(XtX + 1e-8*np.eye(XtX.shape[0])) @ X1.T @ y
            y_pred = X1 @ beta
            ss_tot = np.sum((y - y.mean())**2)
            ss_res = np.sum((y - y_pred)**2)
            r2 = 1.0 - ss_res/(ss_tot + 1e-8)
            score = max(0.0, min(1.0, r2))
        except np.linalg.LinAlgError:
            score = 0.0
    elif method == 'combined':
        score = 0.7*compute_linearity_score_numpy(X,Y,'max_corr') + 0.3*compute_linearity_score_numpy(X,Y,'explained_var')
    else:
        raise ValueError(f"Unknown linearity method: {method}")
    return score

# --- Decision ---

def auto_select_kernel_numpy(X, Y, threshold=0.4, method='combined', verbose=False):
    lin_score = compute_linearity_score_numpy(X,Y,method)
    if lin_score >= threshold:
        rec = 'linear'
        reason = f"linearity_score={lin_score:.3f} ≥ threshold={threshold:.3f}"
        conf = min(1.0, lin_score/0.8)
    else:
        rec = 'rbf'
        reason = f"linearity_score={lin_score:.3f} < threshold={threshold:.3f}"
        conf = min(1.0, (1-lin_score)/0.6)
    info = {
        'linearity_score': lin_score,
        'recommended_kernel': rec,
        'decision_reason': reason,
        'confidence': conf,
        'threshold_used': threshold,
        'method_used': method,
        'data_shape': (X.shape, Y.shape)
    }
    if verbose:
        print("="*48)
        print("Kernel Selection")
        print("="*48)
        print(f"Linearity Score : {lin_score:.4f}")
        print(f"Threshold       : {threshold:.4f}")
        print(f"Recommended     : {rec.upper()}")
        print(f"Reason          : {reason}")
        print(f"Confidence      : {conf:.2f}")
        print("="*48)
    return rec, info

# --- Minimal helper ---

def select_kernel_for_data(X,Y,threshold=0.4):
    return 'linear' if compute_linearity_score_numpy(X,Y,'combined') >= threshold else 'rbf'

# --- Class Interface ---

class KernelSelector:
    def __init__(self, threshold=0.4, method='combined'):
        self.threshold = threshold
        self.method = method
    def select_kernel(self, X, Y, verbose=False):
        return auto_select_kernel_numpy(X, Y, self.threshold, self.method, verbose)
    def check_linearity(self, X, Y):
        return compute_linearity_score_numpy(X, Y, self.method)

if __name__ == '__main__':
    np.random.seed(0)
    X = np.random.randn(500,8)
    Y_lin = X[:,:2] @ np.array([[1.2],[ -0.7]]) + 0.1*np.random.randn(500,1)
    Y_nonlin = np.sin(X[:,0:1]) + 0.1*np.random.randn(500,1)
    sel = KernelSelector()
    print('Linear test:')
    sel.select_kernel(X, Y_lin, verbose=True)
    print('\nNonlinear test:')
    sel.select_kernel(X, Y_nonlin, verbose=True)
