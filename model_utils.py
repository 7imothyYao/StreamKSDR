"""Model utility helpers for feature projection (avoid code duplication).

project_features(model, X):
  Supports BatchKSPCA (expects attributes rff_x, mean_x, U) and OnlineKernelSDR
  Returns centered projected features Phi_x_centered @ U
"""
from __future__ import annotations
import numpy as np

def project_features(model, X: np.ndarray):
    """Project raw input X into model latent space with proper centering.

    Requirements on model:
      - model.rff_x.transform(X) -> Phi_X
      - model has mean_x (vector) computed on training stream
      - model has U (projection matrix)
    """
    if not hasattr(model, 'rff_x') or not hasattr(model, 'U'):
        raise ValueError("Model missing required attributes (rff_x/U).")
    Phi = model.rff_x.transform(X)
    mean_x = getattr(model, 'mean_x', None)
    if mean_x is None:
        # Fallback: compute mean on the fly (rare) – but warn by printing
        mean_x = Phi.mean(axis=0, keepdims=True)
    centered = Phi - mean_x
    return centered @ model.U

__all__ = ["project_features"]
