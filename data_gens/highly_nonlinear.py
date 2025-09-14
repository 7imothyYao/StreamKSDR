import numpy as np


def generate_highly_nonlinear_data(N=2000, noise_level=0.1, random_state=42):
    """Baseline highly nonlinear dataset (10D input, 5D output).

    Mixes spiral / radial components, polynomial and exponential transforms
    to deliver modest dimensionality but clear nonlinear dependencies.
    """
    rng = np.random.RandomState(random_state)

    t = rng.uniform(0, 4 * np.pi, N)
    r = rng.uniform(0.5, 2.0, N)
    z = rng.uniform(-1, 1, N)

    X = np.zeros((N, 10))
    X[:, 0] = r * np.cos(t) + rng.normal(0, noise_level, N)
    X[:, 1] = r * np.sin(t) + rng.normal(0, noise_level, N)
    X[:, 2] = 0.5 * t + rng.normal(0, noise_level, N)
    X[:, 3] = r ** 2 * np.cos(2 * t) + rng.normal(0, noise_level, N)
    X[:, 4] = r ** 2 * np.sin(2 * t) + rng.normal(0, noise_level, N)
    X[:, 5] = t ** 2 * np.sin(t) + rng.normal(0, noise_level, N)
    X[:, 6] = np.exp(-0.5 * r) * np.cos(3 * t) + rng.normal(0, noise_level, N)
    X[:, 7] = np.sin(3 * t) * np.cos(r) + rng.normal(0, noise_level, N)
    X[:, 8] = z * np.tanh(r * np.sin(t)) + rng.normal(0, noise_level, N)
    X[:, 9] = (r - 1) ** 3 + z ** 2 * np.sin(2 * t) + rng.normal(0, noise_level, N)

    Y = np.zeros((N, 5))
    Y[:, 0] = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2) + 0.5 * np.sin(X[:, 2]) + rng.normal(0, noise_level, N)
    Y[:, 1] = np.arctan2(X[:, 1], X[:, 0]) + 0.3 * X[:, 3] + rng.normal(0, noise_level, N)
    Y[:, 2] = X[:, 5] * X[:, 6] + np.cos(X[:, 7]) + rng.normal(0, noise_level, N)
    Y[:, 3] = np.tanh(X[:, 8]) + np.log(1 + X[:, 9] ** 2) + rng.normal(0, noise_level, N)
    Y[:, 4] = np.sin(X[:, 0] + X[:, 1]) * np.exp(-0.1 * (X[:, 3] ** 2 + X[:, 4] ** 2)) + rng.normal(0, noise_level, N)

    return X.astype(np.float32), Y.astype(np.float32)


