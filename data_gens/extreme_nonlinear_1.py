import numpy as np


def generate_extreme_nonlinear_1(N=3000, random_state=42):
    """Extreme nonlinear dataset variant 1 (60D input, 5D output).

    Combines multi-frequency trigonometric components, radial terms and noisy
    harmonic expansions to produce weak linear but strong nonlinear structure.
    """
    rng = np.random.RandomState(random_state)
    t1 = rng.uniform(0, 6 * np.pi, N)
    t2 = rng.uniform(0, 6 * np.pi, N)
    r = rng.uniform(0.2, 3.0, N)

    X = np.zeros((N, 60))
    X[:, 0] = np.sin(t1) + 0.3 * np.sin(5 * t1)
    X[:, 1] = np.cos(t1) + 0.2 * np.cos(7 * t1)
    X[:, 2] = np.sin(t2) * np.cos(t1)
    X[:, 3] = r * np.sin(t1 + t2)
    X[:, 4] = np.exp(-0.3 * r) * np.cos(3 * t2)
    for i in range(5, 60):
        X[:, i] = np.sin((i + 1) * t1 / 10.0) * np.cos((i + 2) * t2 / 12.0) + rng.normal(0, 0.2, N)

    Y = np.zeros((N, 5))
    Y[:, 0] = np.sin(2 * X[:, 0]) + np.cos(3 * X[:, 1]) + 0.5 * np.sign(X[:, 2])
    Y[:, 1] = np.exp(-0.2 * X[:, 3] ** 2) * np.sin(X[:, 0] * X[:, 1])
    Y[:, 2] = X[:, 0] * X[:, 1] * np.tanh(X[:, 2])
    Y[:, 3] = np.sin(8 * X[:, 0]) * np.cos(6 * X[:, 1]) + 0.3 * np.sin(10 * X[:, 2])
    Y[:, 4] = np.sign(X[:, 0]) * np.sign(X[:, 1]) * np.exp(-0.3 * (X[:, 2] ** 2))

    Y += rng.normal(0, 0.2, Y.shape)
    for i in range(Y.shape[1]):
        Y[:, i] = (Y[:, i] - Y[:, i].mean()) / (Y[:, i].std() + 1e-8)

    return X.astype(np.float32), Y.astype(np.float32)


