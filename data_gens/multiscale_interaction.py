import numpy as np

def generate_multiscale_interaction_data(
    n_samples: int = 3000,
    n_latent: int = 6,
    n_segments: int = 5,
    drift_angle: float = 0.35,
    noise_level: float = 0.08,
    random_state: int = 42,
    add_drift: bool = True,
    multi_scale: bool = True,
    n_noise_dims: int = 300,
):
    """Multiscale + mild piecewise drift + high-order interaction dataset.

    Motivation:
    1) Weak linear visibility: overall low Pearson corr between raw X and Y (<~0.3) so linear methods struggle.
    2) Multi-scale: different Y columns depend on both wide (low-frequency) and narrow (local RBF-like) latent structure.
    3) High-order interactions: products, composite sin/cos, radial exp(-r^2/s)*sin(·) terms emphasize nonlinear cross-covariance captured by kernel SDR.
    4) Mild distribution drift: per-segment rotation/scale/shift to showcase potential online adaptation vs full batch retrain.
    5) Distractor noise: many independent Gaussian noise dims reduce SNR to stress supervised kernel selectivity.

    Args:
        n_samples: total samples (temporal order implied)
        n_latent: number of base latent variables (>=4)
        n_segments: number of drift segments
        drift_angle: scale of random rotation per segment
        noise_level: relative noise amplitude per Y column
        random_state: RNG seed
        add_drift: apply piecewise drift if True
        multi_scale: include narrow radial features if True
        n_noise_dims: number of pure noise feature dimensions

    Returns:
        X: (N, D) float32 features
        Y: (N, 5) float32 supervised outputs
    """
    rng = np.random.RandomState(random_state)
    assert n_latent >= 4, "n_latent must be >= 4"
    n_samples = int(n_samples)
    seg_sizes = np.full(n_segments, n_samples // n_segments, dtype=int)
    seg_sizes[: n_samples - seg_sizes.sum()] += 1  # distribute remainder

    # 1. Generate base latent variables (slightly heavy-tailed t-dist)
    Z = rng.standard_t(df=5, size=(n_samples, n_latent)) * 0.9  # 比纯高斯略重尾

    # 2. Piecewise drift: random rotations + scale + shift per segment
    cursor = 0
    for si, size in enumerate(seg_sizes):
        idx = slice(cursor, cursor + size)
        cursor += size
        if add_drift and si > 0:  # 第一段保持原始分布
            # Random chained small rotations + scale + shift
            for a in range(0, n_latent - 1, 2):
                theta = rng.normal(0, drift_angle)
                c, s = np.cos(theta), np.sin(theta)
                Za = Z[idx, a].copy()
                Zb = Z[idx, a + 1].copy()
                Z[idx, a] = c * Za - s * Zb
                Z[idx, a + 1] = s * Za + c * Zb
            scale = rng.uniform(0.85, 1.15, size=(1, n_latent))
            shift = rng.uniform(-0.4, 0.4, size=(1, n_latent)) * (si / max(1, n_segments - 1))
            Z[idx] = Z[idx] * scale + shift

    # 3. Construct multiscale feature blocks from latent pairs
    pairs = []
    for i in range(0, n_latent - 1, 2):
        u = Z[:, i]
        v = Z[:, i + 1]
        r2 = u**2 + v**2
    # Wide-scale and (optionally) narrow-scale radial/angular terms
        feat_block = [
            u,
            v,
            u * v,
            np.sin(u + 0.5 * v),
            np.cos(1.3 * u - 0.7 * v),
            np.exp(-0.5 * r2),  # 宽尺度径向
        ]
        if multi_scale:
            feat_block += [
                np.exp(-3.0 * r2),  # narrow scale (requires small sigma)
                np.sin(2.5 * u) * np.exp(-0.8 * r2),
            ]
        pairs.append(np.column_stack(feat_block))
    X_sig = np.hstack(pairs)

    # Additional cross-pair interactions (raise nonlinear dependence, reduce linear cues)
    if n_latent >= 6:
        u0, v0 = Z[:, 0], Z[:, 1]
        u1, v1 = Z[:, 2], Z[:, 3]
        cross_block = [
            np.tanh(u0 * v1 + u1 * v0),
            np.sin((u0 - u1) * (v0 + 0.5 * v1)),
            np.cos((u0 + u1) * (v0 - v1)),
        ]
        X_sig = np.hstack([X_sig, np.column_stack(cross_block)])

    # 4. Add many independent noise dimensions
    noise = rng.normal(0, 1.0, size=(n_samples, n_noise_dims))

    # 5. Random orthogonal rotation to decorrelate linear traces
    D_sig = X_sig.shape[1]
    Q_rand = rng.normal(0, 1, size=(D_sig, D_sig))
    # Orthogonalize (QR) keep Q
    q, _ = np.linalg.qr(Q_rand)
    X_rot = X_sig @ q
    X = np.hstack([X_rot, noise])

    # 6. Build supervised outputs Y (5 columns) with multiscale + high-order interactions
    a = Z[:, 0]
    b = Z[:, 1]
    c_ = Z[:, 2 % n_latent]
    d = Z[:, 3 % n_latent]
    e = Z[:, 4 % n_latent] if n_latent >= 5 else Z[:, 0]
    f = Z[:, 5 % n_latent] if n_latent >= 6 else Z[:, 1]

    # Y0: quasi-discrete continuous (thresholdable) + smoothing
    raw0 = np.sin(a) + 0.5 * np.cos(b) + 0.3 * np.sin(a * b)
    Y0 = np.tanh(raw0 / 1.3)

    # Y1: wide radial * low-frequency angle
    Y1 = np.exp(-0.4 * (a**2 + b**2)) * np.cos(0.8 * c_) + 0.3 * np.sin(d)

    # Y2: narrow local peak + higher frequency sine
    Y2 = np.exp(-2.5 * (c_**2 + d**2)) * np.sin(2.2 * c_ * d)

    # Y3: composite products + tanh (multi-pair interaction)
    Y3 = np.tanh((a * d) + (b * c_) + 0.5 * (e * f))

    # Y4: mixed scale: wide kernel * narrow kernel * angular modulation
    Y4 = (
        np.exp(-0.5 * (e**2 + f**2))
        * np.exp(-2.0 * (a**2 + 0.5 * b**2))
        * np.sin(a * b + 0.7 * e)
        + 0.15 * np.cos(c_ * f)
    )

    Y = np.column_stack([Y0, Y1, Y2, Y3, Y4])

    # 7. Add noise relative to each column's std
    for j in range(Y.shape[1]):
        s = Y[:, j].std() + 1e-8
        Y[:, j] += rng.normal(0, noise_level * s, size=n_samples)

    return X.astype(np.float32), Y.astype(np.float32)

__all__ = ["generate_multiscale_interaction_data"]
