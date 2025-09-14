import numpy as np


def generate_piecewise_nonlinear(n_samples=2000, random_state=11):
    """Piecewise + discontinuous supervisory signals.

    Constructs targets with conditional branches, sign patterns and localized
    nonlinearities to reduce linear correlation while preserving structured
    nonlinear dependencies suitable for kernel SDR evaluation.
    """
    rng = np.random.RandomState(random_state)
    d = 50
    X = rng.normal(0, 1, (n_samples, d))

    # Core piecewise interactions
    t = X[:, 0] * X[:, 1]
    y0 = np.where(t > 0, np.sin(3 * t), np.cos(2 * t))
    y1 = np.where(X[:, 2] > 0, np.tanh(X[:, 3] - X[:, 4]), -np.tanh(X[:, 3] + X[:, 4]))
    y2 = np.sign(X[:, 5]) * np.exp(-0.2 * (X[:, 6] ** 2))
    y3 = np.where(X[:, 7] + X[:, 8] > 0, np.sin(X[:, 7] * X[:, 8]), np.cos(X[:, 7] - X[:, 8]))
    y4 = np.sign(X[:, 9] * X[:, 10]) * np.sin(5 * X[:, 11])

    Y = np.column_stack([y0, y1, y2, y3, y4])
    Y += rng.normal(0, 0.2, Y.shape)
    for i in range(5):
        Y[:, i] = (Y[:, i] - Y[:, i].mean()) / (Y[:, i].std() + 1e-8)

    return X.astype(np.float32), Y.astype(np.float32)


