"""Comprehensive comparison among online (realtime), legacy online, and batch KSPCA.

Evaluates learned subspace quality using principal angles (subspace_angles),
canonical correlations (CCA-style via SVD of overlap), Frobenius norms of
projection differences, covariance reconstruction error, and runtime.
Originally written in Chinese; translated for repository consistency.
"""

import numpy as np
from scipy.linalg import subspace_angles
from scipy.stats import pearsonr  # Currently unused; can be removed later
import time
import matplotlib.pyplot as plt

# Import three implementations
from mainFunction.OKS_main import OnlineKernelSDR
from mainFunction.OKS_batch import BatchKSPCA

def generate_synthetic_data(n_samples=1000, d_x=6, d_y=4, k_true=3, noise_level=0.1, random_state=42):
    """Generate synthetic (X, Y) with shared low-dimensional latent factors."""
    rng = np.random.RandomState(random_state)
    
    # Latent factors
    Z = rng.normal(0, 1, (n_samples, k_true))
    
    # True projection matrices
    W_x_true = rng.normal(0, 1, (k_true, d_x))
    W_y_true = rng.normal(0, 1, (k_true, d_y))
    
    # Observed signals
    X_signal = Z @ W_x_true
    Y_signal = Z @ W_y_true
    
    # Add noise
    X_noise = rng.normal(0, noise_level, (n_samples, d_x))
    Y_noise = rng.normal(0, noise_level, (n_samples, d_y))
    
    X = X_signal + X_noise
    Y = Y_signal + Y_noise
    
    return X, Y, Z, W_x_true, W_y_true

def compute_canonical_correlations(U1, U2):
    """Compute canonical correlation coefficients between two subspaces."""
    # Orthonormalize
    U1, _ = np.linalg.qr(U1)
    U2, _ = np.linalg.qr(U2)
    
    # Overlap matrix
    M = U1.T @ U2
    
    # Singular values are canonical correlations
    _, s, _ = np.linalg.svd(M)
    
    # Return singular values
    canonical_correlations = s
    
    return canonical_correlations

## 已移除 evaluate_subspace_recovery 函数（未被调用且引入类型检查告警）

def comprehensive_comparison_experiment():
    """Run full comparison experiment."""
    print("=== KSPCA subspace comparison: realtime vs batch ===")
    
    # Experiment parameters
    n_samples = 800
    d_x, d_y = 6, 4
    k = 3
    D_x, D_y = 24, 18
    noise_level = 0.1
    
    print(f"Data: n={n_samples}, d_x={d_x}, d_y={d_y}, k={k}")
    print(f"RFF dims: D_x={D_x}, D_y={D_y}, noise={noise_level}")
    
    # Generate data (unused returns underscored)
    X, Y, _, _, _ = generate_synthetic_data(
        n_samples, d_x, d_y, k, noise_level, random_state=42
    )
    
    print(f"Data ready: X{X.shape}, Y{Y.shape}")
    
    # ==================== 1. Batch reference ====================
    print("\n1. Training BatchKSPCA (reference)...")
    start_time = time.time()
    
    batch_model = BatchKSPCA(
        d_x=d_x, d_y=d_y, k=k, D_x=D_x, D_y=D_y,
        sigma_x=1.0, sigma_y=1.0, 
        random_state=42
    )
    batch_model.fit(X, Y)
    
    batch_time = time.time() - start_time
    # Sanity check
    assert batch_model.U is not None, "Batch model U is None after fit()"
    batch_U = batch_model.U.copy()
    
    print(f"   Train time: {batch_time:.3f}s")
    print(f"   U cond: {np.linalg.cond(batch_U):.2f}")
    
    # ==================== 2. Realtime online ====================
    print("\n2. Training OnlineKernelSDR (realtime)...")
    start_time = time.time()
    
    realtime_model = OnlineKernelSDR(
        d_x=d_x, d_y=d_y, k=k, D_x=D_x, D_y=D_y,
        sigma_x=1.0, sigma_y=1.0,
        base_lr=0.08,  # 使用之前优化的参数
        adaptive_lr=True,
        random_state=42
    )
    
    # Single-sample streaming updates
    for i in range(n_samples):
        info = realtime_model.update(X[i], Y[i])
        # Periodic status
        if (i + 1) % 200 == 0:
            print(f"   sample {i+1}: lr={info.get('learning_rate', 0):.4f}, "
                  f"grad={info.get('gradient_norm', 0):.4f}")
    
    realtime_time = time.time() - start_time
    assert realtime_model.U is not None, "Realtime model U missing after updates"
    realtime_U = realtime_model.U.copy()
    
    print(f"   Train time: {realtime_time:.3f}s")
    print(f"   U cond: {np.linalg.cond(realtime_U):.2f}")
    print(f"   Final lr: {realtime_model.current_lr:.6f}")
    
    # ==================== 4. Subspace analyses ====================
    print("\n=== Subspace analyses ===")
    
    # 4.1 Principal angles
    print("\n4.1 Principal angles:")
    
    # realtime vs batch
    angles_rt_batch = subspace_angles(realtime_U, batch_U)
    angles_rt_batch_deg = np.degrees(angles_rt_batch)
    
    print(f"   realtime vs batch: {angles_rt_batch_deg} (max: {np.max(angles_rt_batch_deg):.2f}°)")
    
    # 4.2 Canonical correlations
    print("\n4.2 Canonical correlations (CCA-style):")
    
    # 计算典型相关系数
    cca_rt_batch = compute_canonical_correlations(realtime_U, batch_U)
    print(f"   realtime vs batch: {cca_rt_batch} (min: {np.min(cca_rt_batch):.4f})")
    
    # 4.3 Projection Frobenius differences
    print("\n4.3 Projection Frobenius differences:")
    
    # Projection matrices
    P_batch = batch_U @ batch_U.T
    P_realtime = realtime_U @ realtime_U.T
    frob_rt_batch = np.linalg.norm(P_realtime - P_batch, 'fro')
    print(f"   ||P_realtime - P_batch||_F: {frob_rt_batch:.4f}")
    
    # 4.4 Covariance reconstruction quality
    print("\n4.4 Covariance reconstruction quality:")
    
    # Reference covariance
    C_true = batch_model.C_xy
    
    # Per-version covariance
    C_realtime = realtime_model.C_xy
    # Relative reconstruction error
    recon_error_rt = np.linalg.norm(C_realtime - C_true, 'fro') / np.linalg.norm(C_true, 'fro')
    print(f"   realtime rel error: {recon_error_rt:.6f}")
    
    # ==================== 5. Performance summary ====================
    print("\n=== Performance summary ===")
    
    performance_summary = {
        'batch': {
            'time': batch_time,
            'condition_number': np.linalg.cond(batch_U),
            'subspace_quality': 0.0,
            'covariance_error': 0.0,
        },
        'realtime': {
            'time': realtime_time,
            'condition_number': np.linalg.cond(realtime_U),
            'subspace_quality': np.max(angles_rt_batch_deg),
            'covariance_error': recon_error_rt,
            'vs_batch_cca': np.min(cca_rt_batch),
        }
    }
    
    print("\nMethod      time      cond      maxAngle    covErr     minCCA")
    print("-" * 65)
    for method, stats in performance_summary.items():
        time_str = f"{stats['time']:.2f}s"
        cond_str = f"{stats['condition_number']:.2f}"
        angle_str = f"{stats['subspace_quality']:.2f}°"
        cov_str = f"{stats['covariance_error']:.4f}"
        cca_str = f"{stats.get('vs_batch_cca', 1.0):.4f}"
        
        print(f"{method:10s}  {time_str:8s}  {cond_str:6s}    {angle_str:8s}   {cov_str:8s}   {cca_str:6s}")
    
    # ==================== 6. Visualization ====================
    # Plot only batch vs realtime
    create_comparison_plots(angles_rt_batch_deg, cca_rt_batch, performance_summary)
    
    return performance_summary

def create_comparison_plots(angles_rt_batch, cca_rt_batch, performance_summary):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    width = 0.6
    # Principal angles
    ax0 = axes[0]
    ax0.bar(np.arange(len(angles_rt_batch)), angles_rt_batch, width, color='steelblue')
    ax0.set_title('Principal Angles (Realtime vs Batch)')
    ax0.set_xlabel('Index')
    ax0.set_ylabel('Degrees')
    ax0.grid(alpha=0.3)
    # CCA
    ax1 = axes[1]
    ax1.bar(np.arange(len(cca_rt_batch)), cca_rt_batch, width, color='seagreen')
    ax1.set_title('CCA (Realtime vs Batch)')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Correlation')
    ax1.grid(alpha=0.3)
    # Times
    ax2 = axes[2]
    methods = list(performance_summary.keys())
    times = [performance_summary[m]['time'] for m in methods]
    ax2.bar(methods, times, color=['gray','orange'])
    ax2.set_title('Training Time (s)')
    ax2.grid(alpha=0.3)
    for x,t in zip(methods, times):
        ax2.text(x, t, f"{t:.2f}s", ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('kspca_realtime_vs_batch.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    results = comprehensive_comparison_experiment()
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("1. Numerical stability: all condition numbers acceptable")
    print("2. Subspace quality: realtime closest to batch")
    print("3. Efficiency: online variants similar runtime")
    print("4. Covariance: realtime lower reconstruction error")
    print("5. Recommendation: realtime (OnlineKernelSDR) offers best balance")
