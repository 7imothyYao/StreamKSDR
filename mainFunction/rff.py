"""Unified Random Fourier Features (RFF) implementation (numpy-only core).

Used by OnlineKernelSDR (OKS_main.py) and BatchKSPCA (OKS_batch.py).
Supports kernels: rbf, laplace, matern32, matern52, linear.

API: RFF(input_dim, output_dim, sigma=1.0, kernel_type='rbf', random_state=None, dtype=np.float32)
Method: transform(x) where x can be (n,d) or (d,) returns (n,D) or (D,)
Linear kernel: output_dim 必须等于 input_dim，直接透传不做随机投影。
"""
import numpy as np

class RFF:
    def __init__(self, input_dim, output_dim, sigma=1.0,
                 kernel_type='rbf', random_state=None,
                 dtype=np.float32):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sigma = float(sigma)
        self.dtype = dtype
        self.kernel = kernel_type.lower()
        self.rng = np.random.RandomState(random_state)
        self.linear = False
        if self.kernel == 'rbf':
            self.W = self.rng.normal(0, 1 / self.sigma, (self.output_dim, input_dim))
        elif self.kernel == 'laplace':
            self.W = self.rng.standard_cauchy((self.output_dim, input_dim)) / self.sigma
        elif self.kernel == 'matern32':
            self.W = self.rng.gamma(1.5, 2 / (self.sigma * np.sqrt(3)), (self.output_dim, input_dim))
        elif self.kernel == 'matern52':
            self.W = self.rng.gamma(2.5, 2 / (self.sigma * np.sqrt(5)), (self.output_dim, input_dim))
        elif self.kernel == 'linear':
            if output_dim != input_dim:
                raise ValueError("For linear kernel, output_dim must equal input_dim (pure passthrough).")
            self.linear = True  # mark identity mapping
            self.W = None
            self.b = None
            self.scale = 1.0
        else:
            # fallback to rbf-like initialization
            self.W = self.rng.normal(0, 1 / self.sigma, (self.output_dim, input_dim))
        if not self.linear:
            self.b = self.rng.uniform(0, 2 * np.pi, self.output_dim)
            self.scale = np.sqrt(2.0 / self.output_dim) if self.output_dim > 0 else 1.0

    def transform(self, x):
        x = np.asarray(x, dtype=self.dtype)
        if self.linear:
            return x  # identity mapping
        # Non-linear kernels: safe to assume self.W and self.b initialized
        linear_proj = x @ self.W.T + self.b  # type: ignore[arg-type]
        return (self.scale * np.cos(linear_proj)).astype(self.dtype)

__all__ = ["RFF"]
