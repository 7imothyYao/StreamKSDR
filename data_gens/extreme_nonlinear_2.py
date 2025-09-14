import numpy as np


def generate_extreme_nonlinear_2(N=2500, random_state=123):
    """Extreme nonlinear dataset variant 2 (80D input) with multiplicative,
    radial decay and composite sinusoidal interactions yielding challenging
    cross-modal covariance patterns for kernel SDR.
    """
    rng = np.random.RandomState(random_state)
    u = rng.uniform(-2, 2, (N, 3))
    v = rng.normal(0, 1, (N, 3))

    # Build high-dim X with multiplicative and radial structure
    X = np.zeros((N, 80))
    X[:, 0] = np.tanh(u[:, 0] * u[:, 1])
    X[:, 1] = np.sin(u[:, 2] * u[:, 0])
    X[:, 2] = np.cos(u[:, 1] * u[:, 2])
    X[:, 3] = np.exp(-0.5 * (u[:, 0] ** 2 + u[:, 1] ** 2))
    X[:, 4] = (u[:, 0] ** 2 - u[:, 1] ** 2) * np.sin(u[:, 2])
    for i in range(5, 80):
        X[:, i] = np.sin(u[:, 0] * (i + 1) / 10.0) * np.cos(u[:, 1] * (i + 2) / 11.0) + 0.1 * v[:, i % 3]

    # Y from mixed nonlinearities
    Y = np.zeros((N, 5))
    Y[:, 0] = np.sin(3 * X[:, 0] + X[:, 1]) + 0.2 * np.sign(X[:, 2])
    Y[:, 1] = np.tanh(X[:, 0] * X[:, 2]) + 0.3 * np.cos(2 * X[:, 4])
    Y[:, 2] = X[:, 0] * X[:, 1] * X[:, 2]
    Y[:, 3] = np.exp(-0.3 * (X[:, 0] ** 2 + X[:, 1] ** 2)) * np.sin(5 * X[:, 2])
    Y[:, 4] = np.sin(4 * X[:, 0]) * np.sin(3 * X[:, 1]) * np.cos(2 * X[:, 2])

    Y += rng.normal(0, 0.25, Y.shape)
    for i in range(Y.shape[1]):
        Y[:, i] = (Y[:, i] - Y[:, i].mean()) / (Y[:, i].std() + 1e-8)

    return X.astype(np.float32), Y.astype(np.float32)


