import numpy as np


def generate_swiss_nonlinear(n_samples=2500, random_state=21):
    """Swiss roll inspired manifold embedded in higher dimensional noisy space.

    Provides curved low-dimensional structure with added harmonics and noise
    to evaluate representation learning and supervised kernel DR.
    """
    rng = np.random.RandomState(random_state)
    t = 1.5 * np.pi * (1 + 2 * rng.rand(n_samples))
    h = 21 * rng.rand(n_samples)
    x = t * np.cos(t)
    y = h
    z = t * np.sin(t)
    base = np.stack([x, y, z], axis=1)

    d = 80
    X = rng.normal(0, 0.1, (n_samples, d))
    X[:, :3] = base + rng.normal(0, 0.1, base.shape)
    for i in range(3, d):
        X[:, i] += np.sin((i + 1) * x / 10.0) + np.cos((i + 2) * z / 9.0) * 0.2

    Y = np.zeros((n_samples, 5))
    Y[:, 0] = np.sin(0.1 * t) + 0.05 * h
    Y[:, 1] = np.sign(np.cos(0.2 * t)) * np.tanh(0.1 * h)
    Y[:, 2] = np.tanh(X[:, 0] * X[:, 2])
    Y[:, 3] = np.exp(-0.01 * (X[:, 0] ** 2 + X[:, 2] ** 2)) * np.cos(0.2 * h)
    Y[:, 4] = np.sin(0.15 * (x + z)) * np.cos(0.1 * (x - z))

    Y += rng.normal(0, 0.15, Y.shape)
    for i in range(5):
        Y[:, i] = (Y[:, i] - Y[:, i].mean()) / (Y[:, i].std() + 1e-8)

    return X.astype(np.float32), Y.astype(np.float32)


