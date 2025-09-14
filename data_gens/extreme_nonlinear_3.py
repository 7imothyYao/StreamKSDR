import numpy as np


def generate_extreme_nonlinear_3(N=2000, random_state=7):
    """Extreme nonlinear dataset variant 3 (100D) based on a torus-like latent
    manifold plus harmonic expansions and localized radial features.
    """
    rng = np.random.RandomState(random_state)

    # Latent manifold on torus-like structure
    theta = rng.uniform(0, 2 * np.pi, N)
    phi = rng.uniform(0, 2 * np.pi, N)

    X = np.zeros((N, 100))
    X[:, 0] = np.cos(theta) * (1 + 0.3 * np.cos(phi))
    X[:, 1] = np.sin(theta) * (1 + 0.3 * np.cos(phi))
    X[:, 2] = 0.3 * np.sin(phi)
    for i in range(3, 100):
        X[:, i] = np.sin((i + 1) * theta / 8.0) + 0.5 * np.cos((i + 2) * phi / 9.0) + rng.normal(0, 0.1, N)

    Y = np.zeros((N, 5))
    Y[:, 0] = np.sin(2 * theta) * np.cos(phi)
    Y[:, 1] = np.sign(np.cos(theta)) * np.sign(np.sin(phi))
    Y[:, 2] = np.tanh(X[:, 0] * X[:, 1])
    Y[:, 3] = np.exp(-0.2 * (X[:, 0] ** 2 + X[:, 1] ** 2)) * np.cos(3 * X[:, 2])
    Y[:, 4] = np.sin(theta + phi) * np.cos(2 * theta - phi)

    Y += rng.normal(0, 0.2, Y.shape)
    for i in range(Y.shape[1]):
        Y[:, i] = (Y[:, i] - Y[:, i].mean()) / (Y[:, i].std() + 1e-8)

    return X.astype(np.float32), Y.astype(np.float32)


