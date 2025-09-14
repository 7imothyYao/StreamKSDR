import numpy as np

def generate_better_xor_data(n_samples=1500, n_features=101, noise_level=0.1, random_state=42):
    """Improved XOR dataset (multi-output) combining:
    - XOR count (discrete-like supervisory signal)
    - Smooth continuous interaction columns
    - Cross-pair interactions to emphasize nonlinear kernel advantage
    """
    rng = np.random.RandomState(random_state)

    # Base XOR latent structure
    n_pairs = 6  # number of XOR pairs
    X = rng.normal(0, 1, (n_samples, n_pairs * 2))

    # 1) XOR count as Y0 (can threshold for classification)
    Y0 = np.sum(((X[:, 0::2] > 0) ^ (X[:, 1::2] > 0)).astype(float), axis=1, keepdims=True)

    # 2) Continuous regression columns (smooth interactions)
    pairs = [(X[:, 2*i], X[:, 2*i+1]) for i in range(n_pairs)]
    Y1 = np.tanh(0.8 * pairs[0][0] * pairs[0][1])          # smoothed XOR-like
    Y2 = np.sin(1.3 * pairs[1][0] * pairs[1][1])           # multi-modal
    Y3 = np.cos(0.9 * pairs[2][0] * pairs[2][1])           # different frequency

    # 3) Cross-pair interaction (nonlinear, kernel-friendly)
    Y4 = np.tanh(0.6 * (pairs[0][0]*pairs[1][1] + pairs[1][0]*pairs[2][1]))

    # 合并所有Y列
    Y = np.column_stack([Y0, Y1, Y2, Y3, Y4]) + rng.normal(0, noise_level, (n_samples, 5))

    # 4) Noise / decoy: add distractor dimensions with variance ~1
    remaining_dims = n_features - n_pairs * 2
    if remaining_dims > 0:
        decoy = rng.normal(0, 1.0, (n_samples, remaining_dims))
        X = np.hstack([X, decoy])

    return X.astype(np.float32), Y.astype(np.float32)
