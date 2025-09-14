import numpy as np

def generate_nuclear_friendly_xor_data(
    n_samples=2000,
    n_latent=8,              # Must be even: number of latent pairs = n_latent//2
    n_noise_dims=200,
    snr_x=6.0,               # Signal-to-noise ratio for X: signal_std / noise_std
    snr_y=6.0,               # Signal-to-noise ratio for each Y column
    random_state=42,
    make_drift=False,        # If True add mild piecewise drift to showcase online adaptation
):
    """Nuclear-friendly XOR dataset (multi-output, mixed discrete + continuous) tailored for
    kernel supervised DR (RBF + OnlineKernelSDR / BatchKSPCA).

    Properties:
    - Intentionally weak linear correlation (XOR + random rotations) but strong nonlinear dependency φ(X)->Y.
    - Multi-output: Y0 quasi-discrete (smoothed count of XOR triggers); Y1–Y4 continuous nonlinear transforms.
    - Kernelize only X while keeping Y amplitude (helps regression comparison).
    - snr_x / snr_y control per-channel noise levels automatically scaled by column std.
    """
    rng = np.random.RandomState(random_state)
    assert n_latent % 2 == 0, "n_latent must be even."

    n_pairs = n_latent // 2

    # 1) Generate latent pairs and apply random 2x2 rotations to break axis-aligned cues
    Z = rng.normal(0, 1, size=(n_samples, 2 * n_pairs))
    X_pairs = []
    parities = []
    radii = []

    for i in range(n_pairs):
        a = Z[:, 2*i]
        b = Z[:, 2*i+1]
    # Random 2x2 rotation
        theta = rng.uniform(0, 2*np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]], dtype=float)
        rot = np.vstack([a, b]).T @ R
        u, v = rot[:, 0], rot[:, 1]
        X_pairs.append(np.column_stack([u, v]))

    # Intermediate quantities feeding Y construction
        parities.append(np.sign(u) * np.sign(v))                  # {-1, +1}
        radii.append(np.sqrt(u**2 + v**2 + 1e-6))

    X_sig = np.hstack(X_pairs)                                    # (N, 2*n_pairs)
    P = np.column_stack(parities)                                 # (N, n_pairs)
    R = np.column_stack(radii)                                    # (N, n_pairs)

    # 2) Build multi-output Y (Y0 discrete-like, Y1..Y4 smooth continuous)
    # Y0: count XOR activations across pairs then smooth / standardize to ~[-1,1]
    XORs = ((X_sig[:, 0::2] > 0).astype(int) ^ (X_sig[:, 1::2] > 0).astype(int)).astype(float)
    Y0 = XORs.sum(axis=1, keepdims=True).astype(float)
    Y0 = np.tanh(0.75 * (Y0 - Y0.mean()) / (Y0.std() + 1e-8))    # 平滑并标准化到 ~[-1,1]

    # Y1~Y4: continuous nonlinear signals highlighting product / phase / radial structure
    # Use first 3 latent pairs (wrap if fewer)
    def pair(i):  # fetch (u,v)
        i = i % n_pairs
        return X_sig[:, 2*i], X_sig[:, 2*i+1]

    u0, v0 = pair(0)
    u1, v1 = pair(1)
    u2, v2 = pair(2)

    Y1 = np.tanh(0.8 * (u0 * v0))                                 # 平滑 XOR 感
    Y2 = np.sin(1.3 * (u1 * v1))                                  # 多峰
    Y3 = np.cos(0.7 * (u2**2 + v2**2)) * np.sign(u2)              # 径向+相位
    Y4 = np.tanh(0.6 * (u0 * v1 + u1 * v2))                       # 跨对交互

    Y = np.column_stack([Y0.ravel(), Y1, Y2, Y3, Y4]).astype(float)

    # 3) Add noise per Y column to match requested snr_y
    for j in range(Y.shape[1]):
        s = Y[:, j].std() + 1e-8
        noise_std = s / float(snr_y)
        Y[:, j] += rng.normal(0, noise_std, size=n_samples)

    # 4) Add noise and distractor (pure noise) dimensions to X per snr_x
    sig_std = X_sig.std(axis=0, keepdims=True) + 1e-8
    X_sig_noisy = X_sig + rng.normal(0, sig_std / float(snr_x), size=X_sig.shape)

    X_noise = rng.normal(0, 1.0, size=(n_samples, n_noise_dims))  # 独立噪声（与 Y 独立）
    X = np.hstack([X_sig_noisy, X_noise]).astype(np.float32)

    # 5) Optional piecewise mild drift to test online adaptation
    if make_drift:
    # Every 500 samples rotate first feature pair slightly + shift one target column
        seg = 500
        for s in range(seg, n_samples, seg):
            X[s:, :2] = X[s:, :2] @ np.array([[np.cos(0.15), -np.sin(0.15)],
                                              [np.sin(0.15),  np.cos(0.15)]])
            # 轻微偏移一个连续目标
            Y[s:, 2] = np.tanh(Y[s:, 2] + 0.1)

    return X.astype(np.float32), Y.astype(np.float32)


def generate_extreme_nuclear_friendly_data(
    n_samples=2000, n_latent=12, n_noise_dims=400, snr_x=8.0, snr_y=8.0, random_state=123, make_drift=True
):
    """
    "极端版"：更多 latent 对、更大噪声维度、更高 SNR，并默认加入分段漂移。
    """
    return generate_nuclear_friendly_xor_data(
        n_samples=n_samples,
        n_latent=n_latent,
        n_noise_dims=n_noise_dims,
        snr_x=snr_x,
        snr_y=snr_y,
        random_state=random_state,
        make_drift=make_drift,
    )
