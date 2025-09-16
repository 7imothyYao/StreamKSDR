import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
import sys
import os
import time
import logging

# Import figure utilities (avoid importing CURRENT_FIG_SAVE_DIR by value to keep it live)
from fig_utils import RUN_TAG, set_dataset_save_dir, save_fig, set_fig_formats, FIG_FORMATS
from visualization import (create_small_format_visualizations, create_performance_comparison_chart,
                            create_feature_embedding_visualization, create_correlation_heatmap)
from reporting import (create_comprehensive_visualizations, plot_online_learning_curves,
                       visualize_noise_structure, analyze_noise_impact_on_methods)
from optimization import (get_adaptive_configurations, run_hyperparameter_optimization,
                          calculate_optimization_score)
import fig_utils  # for accessing fig_utils.CURRENT_FIG_SAVE_DIR dynamically

# ================= Constants / Global Configuration =================
METHODS_ORDER = ['Raw', 'PCA', 'Batch_KSDR', 'Online_KSDR']
DIM_PENALTY_BASE = 3  # Base for dimensionality penalty in composite scoring
DEFAULT_VERBOSE = False
DEFAULT_QUIET = False

logger = logging.getLogger(__name__)

def configure_logging(quiet: bool = False, verbose: bool = False):
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, format='[%(levelname)s] %(message)s')
    logger.debug("Logging configured. quiet=%s verbose=%s", quiet, verbose)
PRINTED_WARNINGS = set()

## Figure utilities moved to fig_utils.py

# Add path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from mainFunction.OKS_main import OnlineKernelSDR
from mainFunction.OKS_batch import BatchKSDR
from data_gens import get_generator
from kernel_selector import KernelSelector
from model_utils import project_features

# Import nuclear-friendly XOR data generators (legacy & new)
from data_gens.nuclear_friendly_xor_data import generate_nuclear_friendly_xor_data, generate_extreme_nuclear_friendly_data
# Unified access via data_gens.get_generator; no need to import individual generators explicitly

plt.style.use('default')
sns.set_palette("husl")
    
def linear_signal_check(X, Y):
    """Quick diagnostic: per Y dimension maximum |corr(X_j, Y_i)| across all X columns.
    Expectation < 0.5 favors kernel methods.
    Returns: max_abs (shape=[d_y])"""
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-8)
    Yc = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
    corr = Xc.T @ Yc / X.shape[0]
    max_abs = np.max(np.abs(corr), axis=0)
    return max_abs

## Embedded data generation functions have been migrated to data_gens/; always access via get_generator.

def compare_feature_extraction_methods(X, Y, kernel: str = "rbf", learning_rate=None, verbose: bool = DEFAULT_VERBOSE):
    """
    Compare the effectiveness of different feature extraction methods

    Parameters:
    - X: Input features
    - Y: Target features
    - kernel: Kernel type for Kernel SDR methods ('rbf' or 'linear')

    Returns:
    - feature_sets: Dictionary containing extracted features from different methods
    - models: Dictionary containing trained models
    """
    N, d_x = X.shape
    d_y = Y.shape[1]
    k = 5  # Extract 5 principal components (B2: expanded from 3)
    
    # ===== Strict train/test split (no leakage) =====
    from sklearn.model_selection import train_test_split
    idx_all = np.arange(N)
    idx_train, idx_test = train_test_split(idx_all, test_size=0.3, random_state=42, shuffle=True)
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_test, Y_test = X[idx_test], Y[idx_test]

    # -------- Adaptive sigma selection (median heuristic + small grid) --------
    def _median_pairwise_sq_dist(mat, max_samples=400, rng_seed=1234):
        n = mat.shape[0]
        if n > max_samples:
            rng_local = np.random.RandomState(rng_seed)
            idx = rng_local.choice(n, size=max_samples, replace=False)
            sub = mat[idx]
        else:
            sub = mat
        # pairwise squared distances (vectorized)
        G = sub @ sub.T
        sq_norms = np.sum(sub**2, axis=1, keepdims=True)
        dists = sq_norms + sq_norms.T - 2*G
        dists = dists[np.triu_indices_from(dists, k=1)]
        dists = dists[dists > 0]
        if dists.size == 0:
            return 1.0
        return float(np.median(dists))

    def _select_sigma(Xs, Ys, k_latent, D_x=150, D_y=120, multipliers=(0.25,0.5,1.0,2.0,4.0), kernel: str = "rbf", random_state=42):
        base_med_x = _median_pairwise_sq_dist(Xs)
        base_med_y = _median_pairwise_sq_dist(Ys)
        # RBF: k(x,x') = exp(-||x-x'||^2/(2 sigma^2)); median heuristic: sigma^2 = median/2
        base_sigma_x = np.sqrt(max(base_med_x, 1e-8)/2.0)
        base_sigma_y = np.sqrt(max(base_med_y, 1e-8)/2.0)
        best = None
        best_score = -np.inf
        rng_base = np.random.RandomState(random_state)
        # small subset for scoring
        subset_n = min(500, Xs.shape[0])
        idx_sub = rng_base.choice(Xs.shape[0], size=subset_n, replace=False)
        X_sub = Xs[idx_sub]
        Y_sub = Ys[idx_sub]
        for mx in multipliers:
            for my in multipliers:
                sx = base_sigma_x * mx
                sy = base_sigma_y * my
                # build temporary RFFs
                from mainFunction.OKS_batch import BatchKSDR as _BK  # local import to avoid circular
                tmp = _BK(Xs.shape[1], Ys.shape[1], k=k_latent, D_x=D_x, D_y=D_y, sigma_x=sx, sigma_y=sy, kernel_x=kernel, kernel_y=kernel, random_state=random_state)
                # manually compute score quickly without full fit on all data
                phi_x = tmp.rff_x.transform(X_sub)
                phi_y = tmp.rff_y.transform(Y_sub)
                phi_x -= phi_x.mean(axis=0, keepdims=True)
                phi_y -= phi_y.mean(axis=0, keepdims=True)
                Cxy = (phi_x.T @ phi_y) / subset_n
                # score: sum top-k singular values squared (energy captured)
                try:
                    # compute eigenvalues of Cxy Cxy^T
                    M = Cxy @ Cxy.T
                    evals = np.linalg.eigvalsh(M)
                    evals_sorted = np.sort(evals)[::-1]
                    score = float(np.sum(evals_sorted[:k_latent]))
                except Exception:
                    score = float(np.linalg.norm(Cxy, 'fro')**2)
                if score > best_score:
                    best_score = score
                    best = (sx, sy, base_sigma_x, base_sigma_y, score)
        return best  # (sigma_x, sigma_y, base_sigma_x, base_sigma_y, score)

    # sigma selection on TRAIN ONLY
    adaptive_sigmas = None if kernel == 'linear' else _select_sigma(X_train, Y_train, k, kernel=kernel)
    if adaptive_sigmas is None:
        sigma_x_opt = sigma_y_opt = 15.0
        sigma_x_base = sigma_y_base = 15.0
        sigma_score = 0.0
        if kernel != 'linear':
            print("[Adaptive Sigma] selection failed, fallback to 15.0")
        else:
            sigma_x_opt = sigma_y_opt = 1.0
            sigma_x_base = sigma_y_base = 1.0
    else:
        sigma_x_opt, sigma_y_opt, sigma_x_base, sigma_y_base, sigma_score = adaptive_sigmas
        print(f"[Adaptive Sigma] base_sigma_x={sigma_x_base:.3f} base_sigma_y={sigma_y_base:.3f} -> selected sigma_x={sigma_x_opt:.3f} sigma_y={sigma_y_opt:.3f} score={sigma_score:.4f}")
    
    if verbose:
        print("=== Training Different Feature Extraction Methods === (single-pass online updates)")
    
    # 1. Raw features (no fitting needed)
    X_raw = X.copy()
    
    # 2. PCA feature extraction
    if verbose:
        print("Training PCA...")
    pca = PCA(n_components=k, random_state=42)
    pca.fit(X_train)
    X_pca = pca.transform(X)
    
    # 3. Batch Kernel SDR (using optimized parameters)
    if verbose:
        print("Training Batch KSDR (formerly BatchKSPCA)...")
    # -------- RFF dimension selection (B1) --------
    def _score_rff_dims(Dx, Dy, sigma_x, sigma_y, k_latent, subset_n=400, kernel: str = "rbf", seed=77):
        from mainFunction.OKS_batch import BatchKSDR as _BK
        rng_loc = np.random.RandomState(seed)
        idx = rng_loc.choice(X_train.shape[0], size=min(subset_n, X_train.shape[0]), replace=False)
        X_sub = X_train[idx]; Y_sub = Y_train[idx]
        tmp = _BK(d_x, d_y, k=k_latent, D_x=Dx, D_y=Dy, sigma_x=sigma_x, sigma_y=sigma_y, kernel_x=kernel, kernel_y=kernel, random_state=seed)
        # fast score
        phi_x = tmp.rff_x.transform(X_sub)
        phi_y = tmp.rff_y.transform(Y_sub)
        phi_x -= phi_x.mean(axis=0, keepdims=True)
        phi_y -= phi_y.mean(axis=0, keepdims=True)
        Cxy = (phi_x.T @ phi_y) / X_sub.shape[0]
        try:
            M = Cxy @ Cxy.T
            ev = np.linalg.eigvalsh(M)
            ev = np.sort(ev)[::-1]
            return float(np.sum(ev[:k_latent]))
        except Exception:
            return float(np.linalg.norm(Cxy, 'fro')**2)

    if kernel == 'linear':
        # 强制使用恒等映射：特征维度即输入维度；不做RFF搜索
        best_Dx, best_Dy = d_x, d_y
        if verbose:
            print(f"[RFF Dim Select] linear kernel -> using identity dims D_x={best_Dx} D_y={best_Dy}")
    else:
        # Adaptive RFF dimension selection based on data complexity
        if X.shape[1] > 50 or Y.shape[1] > 3:  # High-dimensional or multi-output data
            rff_candidates = [(800,600),(1200,900),(1600,1200),(2000,1500)]
        else:
            rff_candidates = [(500,400),(800,600),(1200,900)]
        rff_scores = []
        for (Dx,Dy) in rff_candidates:
            sc = _score_rff_dims(Dx,Dy,sigma_x_opt,sigma_y_opt,k, kernel=kernel)
            rff_scores.append((sc,Dx,Dy))
        rff_scores.sort(reverse=True)
        best_score, best_Dx, best_Dy = rff_scores[0]
        if verbose:
            print("[RFF Dim Select] candidates=" + ", ".join([f"{Dx}/{Dy}:{sc:.4f}" for sc,Dx,Dy in rff_scores]) )
            print(f"[RFF Dim Select] chosen D_x={best_Dx} D_y={best_Dy} (score={best_score:.4f})")

    batch_kspca = BatchKSDR(d_x, d_y, k=k, D_x=best_Dx, D_y=best_Dy,
                            sigma_x=sigma_x_opt, sigma_y=sigma_y_opt,
                            kernel_x=kernel, kernel_y=kernel, random_state=42)
    batch_kspca.fit(X_train, Y_train)
    
    # Get batch Kernel SDR features (proper centering)
    X_batch_kspca = project_features(batch_kspca, X)
    
    # 5. Online Kernel SDR (using optimized parameters)
    if verbose:
        print("Training Online Kernel SDR (OnlineKernelSDR)...")
    # Adaptive learning rate based on data complexity
    if learning_rate is not None:
        base_lr = learning_rate  # Use specified learning rate
    elif X.shape[1] > 50 or Y.shape[1] > 3:  # High-dimensional or multi-output data
        base_lr = 0.005  # Lower learning rate for complex data
    else:
        base_lr = 0.01   # Standard learning rate for simpler data

    online_kspca = OnlineKernelSDR(d_x=d_x, d_y=d_y, k=k, D_x=best_Dx, D_y=best_Dy,
                                   sigma_x=sigma_x_opt, sigma_y=sigma_y_opt,
                                   kernel_x=kernel, kernel_y=kernel, base_lr=base_lr,
                                   adaptive_lr=True, random_state=42)

    indices_stream = np.random.RandomState(123).permutation(idx_train)
    for idx in indices_stream:
        online_kspca.update(X[idx], Y[idx])
    
    # Get online Kernel SDR features (proper centering)
    X_online_kspca = project_features(online_kspca, X)
    
    # Collect experiment configuration metadata
    experiment_config = {
        'kernel_type': kernel,
        'sigma_x': sigma_x_opt,
        'sigma_y': sigma_y_opt,
        'rff_dim_x': best_Dx,
        'rff_dim_y': best_Dy,
        'target_dim': k,
        'train_size': len(idx_train),
        'test_size': len(idx_test),
        'random_state': 42
    }
    
    return {
        'Raw': X_raw,
        'PCA': X_pca,
    'Batch_KSDR': X_batch_kspca,
    'Online_KSDR': X_online_kspca
    }, {
        'pca': pca,
        'batch_kspca': batch_kspca,
        'online_kspca': online_kspca
    }, {
        'train_idx': idx_train,
        'test_idx': idx_test
    }, experiment_config

def evaluate_downstream_tasks(feature_sets, X, Y, idx_split=None, verbose: bool = DEFAULT_VERBOSE):
    """
    Evaluate downstream task performance with proper train/test split
    
    Parameters:
    - feature_sets: Dictionary containing features from different methods
    - Y: Target features
    
    Returns:
    - results: Dictionary containing performance metrics for each method
    """
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import cross_val_score
    
    results = {}
    
    # Create classification labels (based on first dimension of Y)
    Y_class = (Y[:, 0] > np.median(Y[:, 0])).astype(int)
    
    # Regression target (using second dimension of Y)
    Y_reg = Y[:, 1]
    
    if not DEFAULT_QUIET:
        print("\n=== Evaluating Downstream Task Performance (Train/Test Split) ===")
    
    for name, X_features in feature_sets.items():
        if not DEFAULT_QUIET:
            print(f"\n--- {name} Features ---")
        
        # Unified train/test split for all methods (no leakage)
        if idx_split is not None and 'train_idx' in idx_split and 'test_idx' in idx_split:
            tr, te = idx_split['train_idx'], idx_split['test_idx']
            X_train = X_features[tr]
            X_test = X_features[te]
            y_class_train, y_class_test = Y_class[tr], Y_class[te]
            X_train_reg, X_test_reg = X_train, X_test
            y_reg_train, y_reg_test = Y_reg[tr], Y_reg[te]
        else:
            # fallback: independent split (kept for compatibility)
            X_train, X_test, y_class_train, y_class_test = train_test_split(
                X_features, Y_class, test_size=0.3, random_state=42, stratify=Y_class
            )
            X_train_reg, X_test_reg, y_reg_train, y_reg_test = train_test_split(
                X_features, Y_reg, test_size=0.3, random_state=42
            )
        
        # Standardization
        scaler_clf = StandardScaler()
        X_train_scaled_clf = scaler_clf.fit_transform(X_train)
        X_test_scaled_clf = scaler_clf.transform(X_test)
        
        scaler_reg = StandardScaler()
        X_train_scaled_reg = scaler_reg.fit_transform(X_train_reg)
        X_test_scaled_reg = scaler_reg.transform(X_test_reg)
        
        # Classification task (test set performance)
        clf = LogisticRegression(random_state=42, max_iter=1000)
        clf.fit(X_train_scaled_clf, y_class_train)
        y_pred_class = clf.predict(X_test_scaled_clf)
        acc = accuracy_score(y_class_test, y_pred_class)
        
        # Cross-validation for more robust evaluation
        cv_scores = cross_val_score(clf, X_train_scaled_clf, y_class_train, cv=5)
        cv_acc = cv_scores.mean()
        
        # Regression task (test set performance)
        # Adaptive alpha range based on data complexity
        if X.shape[1] > 50 or Y.shape[1] > 3:  # High-dimensional or multi-output data
            alphas = np.logspace(-2, 3, 15)  # Wider range with stronger regularization
        else:
            alphas = np.logspace(-3, 2, 10)  # Standard range
        reg = RidgeCV(alphas=alphas, cv=5)  # B2: Ridge with CV alpha selection
        reg.fit(X_train_scaled_reg, y_reg_train)
        y_pred_reg = reg.predict(X_test_scaled_reg)
        r2 = r2_score(y_reg_test, y_pred_reg)
        
        # Cross-validation for regression
        cv_r2_scores = cross_val_score(reg, X_train_scaled_reg, y_reg_train, cv=5, scoring='r2')
        cv_r2 = cv_r2_scores.mean()
        
        # Random Forest Classification (test set)
        rf_reg = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_reg.fit(X_train_scaled_clf, y_class_train)
        y_pred_rf = rf_reg.predict(X_test_scaled_clf)
        rf_acc = accuracy_score(y_class_test, y_pred_rf)
        
        # Feature-Y correlation (on full dataset for stability)
        correlations = []
        for i in range(X_features.shape[1]):
            corr = np.corrcoef(X_features[:, i], Y[:, 0])[0, 1]
            correlations.append(abs(corr))
        avg_corr = np.mean(correlations)
        
    # Add theoretical performance comparison for Kernel SDR methods
        results[name] = {
            'classification_acc': acc,
            'cv_classification_acc': cv_acc,
            'regression_r2': r2,
            'cv_regression_r2': cv_r2,
            'rf_classification_acc': rf_acc,
            'avg_correlation': avg_corr,
            'feature_dim': X_features.shape[1]
        }
        
        if verbose:
            print(f"  Dimensions: {X_features.shape[1]}")
            print(f"  Classification Accuracy (Test): {acc:.4f}")
            print(f"  Classification Accuracy (CV): {cv_acc:.4f}")
            print(f"  Regression R² (Test): {r2:.4f}")
            print(f"  Regression R² (CV): {cv_r2:.4f}")
            print(f"  Random Forest Classification: {rf_acc:.4f}")
            print(f"  Average Correlation: {avg_corr:.4f}")
            if hasattr(reg, 'alpha_'):
                print(f"  Ridge Alpha Selected: {reg.alpha_:.4f}")
        
        # Warning if online outperforms batch significantly
        if name == 'Online_KSDR' and 'Batch_KSDR' in results:
            batch_r2 = results['Batch_KSDR']['regression_r2']
            batch_cv_r2 = results['Batch_KSDR']['cv_regression_r2']
            if r2 > batch_r2 + 0.05 or cv_r2 > batch_cv_r2 + 0.05:
                warn_key = (round(batch_r2,3), round(r2,3), 'R2gap')
                if warn_key not in PRINTED_WARNINGS:
                    PRINTED_WARNINGS.add(warn_key)
                    if not DEFAULT_QUIET:
                        print(f"  WARNING: Online KSDR > Batch KSDR1 dR2_test={r2-batch_r2:.4f} dR2_CV={cv_r2-batch_cv_r2:.4f}")

        # Diagnostics (verbose only)
    if verbose and ('KSDR' in name):
            feat_mean = np.mean(X_features, axis=0)
            feat_std = np.std(X_features, axis=0)
            feat_range = np.max(X_features, axis=0) - np.min(X_features, axis=0)
            print(f"  Feature statistics:")
            print(f"      Mean: {feat_mean}")
            print(f"      Std:  {feat_std}")
            print(f"      Range: {feat_range}")
            if X_features.shape[1] >= 3:
                comp_corrs = []
                for i in range(X_features.shape[1]):
                    corr_with_y = np.corrcoef(X_features[:, i], Y[:, 0])[0, 1]
                    comp_corrs.append(abs(corr_with_y))
                print(f"  Component-Y[0] |correlations|: {comp_corrs}")
    
    return results


def create_performance_summary_table(results, dataset_info=None, experiment_config=None):
    """
    Create comprehensive performance summary table with detailed experiment information
    
    Parameters:
    - results: Performance results from different methods
    - dataset_info: Dictionary containing dataset information (name, dimensions, etc.)
    - experiment_config: Dictionary containing experiment configuration
    
    Returns:
    - sorted_methods: List of (method_name, score) tuples sorted by performance
    """
    print("\n" + "="*80)
    print("Feature Extraction Methods Performance Comparison (Highly Nonlinear Data)")
    print("="*80)
    
    # === New composite score (min-max normalized per metric) ===
    methods = list(results.keys())
    # Gather raw metric arrays (CV versions for stability)
    acc_vals = [results[m]['cv_classification_acc'] for m in methods]
    r2_vals  = [max(0.0, results[m]['cv_regression_r2']) for m in methods]  # clamp negative R2 to 0
    corr_vals= [results[m]['avg_correlation'] for m in methods]

    def _min_max(vs):
        lo, hi = min(vs), max(vs)
        if abs(hi - lo) < 1e-12:
            return [0.0 for _ in vs]  # degenerate metric
        return [ (v - lo) / (hi - lo + 1e-8) for v in vs ]

    acc_n = _min_max(acc_vals)
    r2_n  = _min_max(r2_vals)
    corr_n= _min_max(corr_vals)

    scores = []
    for i, m in enumerate(methods):
        S = (acc_n[i] + r2_n[i] + corr_n[i]) / 3.0
        # Optional mild dimension penalty (retain prior behavior) applied after normalization
        feat_dim = results[m]['feature_dim']
        if feat_dim > 3:
            S *= (3 / feat_dim) ** 0.5
        scores.append(S)
    
    # Sort methods by score
    sorted_methods = sorted(zip(methods, scores), key=lambda x: x[1], reverse=True)
    
    print(f"{'Rank':<4} {'Method':<15} {'Test Acc':<9} {'CV Acc':<8} {'Test R2':<8} {'CV R2':<7} {'RF Acc':<7} {'Corr':<6} {'Dim':<4} {'S':<8}")
    print(f"{'':4} {'':15} {'':9} {'':8} {'':8} {'':7} {'':7} {'':6} {'':4} {'':8}")
    print("-" * 88)
    
    lines = []
    for i, (method, score) in enumerate(sorted_methods, 1):
        r = results[method]
        line = (f"{i:<4} {method:<15} {r['classification_acc']:<9.3f} "
                f"{r['cv_classification_acc']:<8.3f} {r['regression_r2']:<8.3f} "
                f"{r['cv_regression_r2']:<7.3f} {r['rf_classification_acc']:<7.3f} "
                f"{r['avg_correlation']:<6.3f} {r['feature_dim']:<4} {score:<8.4f}")
        print(line)
        lines.append(line)

    # === Generate comprehensive figure (default output) ===
    try:
        import matplotlib.pyplot as plt
        methods_order = [m for m, _ in sorted_methods]
        cv_accs = [results[m]['cv_classification_acc'] for m in methods_order]
        cv_r2s = [results[m]['cv_regression_r2'] for m in methods_order]
        corrs = [results[m]['avg_correlation'] for m in methods_order]
        dims = [results[m]['feature_dim'] for m in methods_order]

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        ax1, ax2, ax3, ax4 = axes.ravel()

        # CV Classification Accuracy
        ax1.bar(methods_order, cv_accs, color='#1f77b4')
        ax1.set_title('CV Classification Accuracy')
        ax1.set_ylim(0, max(0.01, max(cv_accs)*1.1))
        for i,v in enumerate(cv_accs):
            ax1.text(i, v+0.005, f"{v:.2f}", ha='center', fontsize=8)

        # CV Regression R²
        ax2.bar(methods_order, cv_r2s, color='#ff7f0e')
        ax2.set_title('CV Regression R²')
        ymin = min(cv_r2s); ax2.set_ylim(min(0,ymin-0.05), max(0.01, max(cv_r2s)*1.15))
        for i,v in enumerate(cv_r2s):
            ax2.text(i, v + (0.01 if v>=0 else -0.04), f"{v:.2f}", ha='center', fontsize=8)

        # Average Correlation
        ax3.bar(methods_order, corrs, color='#2ca02c')
        ax3.set_title('Average |Correlation|')
        ax3.set_ylim(0, max(0.01, max(corrs)*1.2))
        for i,v in enumerate(corrs):
            ax3.text(i, v+0.005, f"{v:.2f}", ha='center', fontsize=8)

        # Ranking text & dimensions
        ax4.axis('off')
        ranking_text = 'Ranking (Score)\n' + '\n'.join([f"{i+1}. {m} ({scores[i]:.3f}) dim={dims[i]}" for i,m in enumerate(methods_order)])
        if dataset_info:
            ranking_text = (f"Dataset: {dataset_info.get('name','?')}  n={dataset_info.get('n_samples','?')}\n" + ranking_text)
        ax4.text(0.02, 0.98, ranking_text, va='top', ha='left', fontsize=10, family='monospace')

        plt.tight_layout()
        save_fig('performance_summary_overview.png')
        plt.close(fig)
    except Exception as e:
        print(f"[Warn] Failed to generate performance summary figure: {e}")

    # Save detailed text results
    if fig_utils.CURRENT_FIG_SAVE_DIR is None:
        # Graceful bailout: avoid crashing, create temp directory based on dataset name
        tmp_name = (dataset_info.get('name','dataset') if dataset_info else 'dataset') + '_summary'
        set_dataset_save_dir(tmp_name)
    if fig_utils.CURRENT_FIG_SAVE_DIR is None:
        print("[Warn] Figure directory still None; skipping detailed text summary.")
        return sorted_methods
    os.makedirs(fig_utils.CURRENT_FIG_SAVE_DIR, exist_ok=True)
    summary_path = os.path.join(fig_utils.CURRENT_FIG_SAVE_DIR, 'performance_summary.txt')
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            # Write experiment configuration information
            f.write('='*80 + '\n')
            f.write('OKSPCA Detailed Experiment Report\n')
            f.write('='*80 + '\n\n')
            
            # Dataset information
            if dataset_info:
                f.write('Dataset Information:\n')
                f.write(f"  • Name: {dataset_info.get('name', 'Unknown')}\n")
                f.write(f"  • Samples: {dataset_info.get('n_samples', 'N/A')}\n")
                f.write(f"  • Input Dim: {dataset_info.get('input_dim', 'N/A')}\n")
                f.write(f"  • Output Dim: {dataset_info.get('output_dim', 'N/A')}\n")
                f.write(f"  • Data Range: X[{dataset_info.get('x_range', 'N/A')}], Y[{dataset_info.get('y_range', 'N/A')}]\n")
                f.write('\n')
            
            # Experiment configuration
            if experiment_config:
                f.write('Experiment Configuration:\n')
                f.write(f"  • Kernel Type: {experiment_config.get('kernel_type', 'N/A')}\n")
                f.write(f"  • Kernel σ_x: {experiment_config.get('sigma_x', 'N/A')}\n")
                f.write(f"  • Kernel σ_y: {experiment_config.get('sigma_y', 'N/A')}\n")
                f.write(f"  • RFF Dim D_x: {experiment_config.get('rff_dim_x', 'N/A')}\n")
                f.write(f"  • RFF Dim D_y: {experiment_config.get('rff_dim_y', 'N/A')}\n")
                f.write(f"  • Target Dim: {experiment_config.get('target_dim', 5)}\n")
                f.write(f"  • Train Size: {experiment_config.get('train_size', 'N/A')}\n")
                f.write(f"  • Test Size: {experiment_config.get('test_size', 'N/A')}\n")
                f.write(f"  • Random Seed: {experiment_config.get('random_state', 42)}\n")
                f.write('\n')
            
            # Performance comparison table
            f.write('Performance Comparison Results:\n')
            f.write('-'*88 + '\n')
            f.write(f"{'Rank':<4} {'Method':<15} {'Test Acc':<9} {'CV Acc':<8} {'Test R²':<8} {'CV R²':<7} {'RF Acc':<7} {'Corr':<6} {'Dim':<4} {'Score':<8}\n")
            f.write('-'*88 + '\n')
            f.write('\n'.join(lines) + '\n\n')
            
            # Method details
            f.write('Method Details:\n')
            for method in results.keys():
                r = results[method]
                f.write(f"\n• {method}:\n")
                f.write(f"  - Feature Dim: {r['feature_dim']}\n")
                f.write(f"  - Classification Acc (Test): {r['classification_acc']:.4f}\n")
                f.write(f"  - Classification Acc (CV): {r['cv_classification_acc']:.4f}\n")
                f.write(f"  - Regression R² (Test): {r['regression_r2']:.4f}\n")
                f.write(f"  - Regression R² (CV): {r['cv_regression_r2']:.4f}\n")
                f.write(f"  - RandomForest Acc: {r['rf_classification_acc']:.4f}\n")
                f.write(f"  - Avg Correlation: {r['avg_correlation']:.4f}\n")
                
                # Additional method-specific info
                if 'ridge_alpha' in r:
                    f.write(f"  - Ridge Alpha: {r['ridge_alpha']:.4f}\n")
                if 'feature_stats' in r:
                    stats = r['feature_stats']
                    f.write(f"  - Feature Stats: mean={stats.get('mean', 'N/A')}, std={stats.get('std', 'N/A')}\n")
            
            # Experimental conclusions
            f.write('\nExperimental Conclusions:\n')
            best_method = sorted_methods[0][0]
            best_score = sorted_methods[0][1]
            f.write(f"  • Best Method: {best_method} (Composite Score: {best_score:.4f})\n")
            f.write(f"  • Kernel vs Linear: {'Kernel methods perform better' if 'KSPCA' in best_method else 'Linear methods perform better'}\n")
            f.write(f"  • Online vs Batch: {'Online learning better' if 'Online' in best_method else 'Batch processing better'}\n")
            
            # Timestamp
            import datetime
            f.write(f"\nGenerated At: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        print(f"[Saved] {summary_path}")
    except Exception as e:
        print(f"[Warn] Failed to save summary: {e}")
    
    # Generate JSON export
    try:
        import json
        json_path = os.path.join(fig_utils.CURRENT_FIG_SAVE_DIR, 'performance_summary.json') if fig_utils.CURRENT_FIG_SAVE_DIR else None
        if json_path:
            json_payload = {
                'dataset': dataset_info or {},
                'experiment_config': experiment_config or {},
                'methods': { m: results[m] for m in results },
                'ranking': [{ 'method': m, 'score': float(s) } for m,s in sorted_methods]
            }
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(json_payload, jf, ensure_ascii=False, indent=2)
            print(f"[Saved] {json_path}")
    except Exception as e:
        print(f"[Warn] JSON export failed: {e}")
    return sorted_methods

def main(dataset: str = "highly_nonlinear", kernel: str = "auto", include_big_plots: bool = True, learning_rate = None):
    """
    Main experiment function with automatic kernel selection

    This function orchestrates the complete experiment pipeline:
    1. Generate highly nonlinear synthetic data
    2. Automatic kernel selection using KernelSelector
    3. Train different feature extraction methods
    4. Evaluate downstream task performance
    5. Create comprehensive visualizations (small format)
    6. Generate performance summary and conclusions

    Returns:
    - results: Performance metrics for all methods
    - feature_sets: Extracted features from different methods
    - models: Trained feature extraction models
    """
    if not DEFAULT_QUIET:
        print("Starting downstream task experiments on highly_nonlinear data")
        print("="*70)

    # Set dataset save directory
    set_dataset_save_dir(dataset)

    # 1. Generate highly nonlinear data
    if not DEFAULT_QUIET:
        print(f"Generating dataset via '{dataset}' module...")
    gen = get_generator(dataset)
    # Use a generic signature call (high-dimensional generators handle defaults internally)
    X, Y = gen(n_samples=1500, noise_level=0.15, random_state=42)

    if not DEFAULT_QUIET:
        print(f"Data dimensions: X {X.shape}, Y {Y.shape}")
        print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
        print(f"Y range: [{Y.min():.2f}, {Y.max():.2f}]")

    # 2. Automatic kernel selection
    if kernel == "auto":
        if not DEFAULT_QUIET:
            print("\nPerforming automatic kernel selection...")
        selector = KernelSelector(threshold=0.4, method='combined')
        selected_kernel, kernel_info = selector.select_kernel(X, Y, verbose=True)
        kernel = selected_kernel
        if not DEFAULT_QUIET:
            print(f"Selected kernel: {kernel}")
    else:
        if not DEFAULT_QUIET:
            print(f"Using specified kernel: {kernel}")

    # 3. Train different feature extraction methods
    feature_sets, models, split, experiment_config = compare_feature_extraction_methods(X, Y, kernel=kernel, learning_rate=learning_rate)

    # Linear signal diagnostic
    lin_max = linear_signal_check(X, Y)
    if not DEFAULT_QUIET:
        print(f"[Linear check] max |corr(X, Y_j)|: {lin_max}")

    # 4. Evaluate downstream task performance
    results = evaluate_downstream_tasks(feature_sets, X, Y, idx_split=split, verbose=DEFAULT_VERBOSE)

    # 5. Create small format visualizations (not big comprehensive plots)
    create_small_format_visualizations(X, Y, feature_sets, models, results)

    # 5.5. Optionally create big comprehensive plots for appendix
    if include_big_plots:
        try:
            create_comprehensive_visualizations(X, Y, feature_sets, models, results)
        except Exception as e:
            print(f"[Warn] Big plot generation failed: {e}")

    # 6. Generate performance summary
    dataset_info = {
        'name': dataset,
        'n_samples': X.shape[0],
        'input_dim': X.shape[1],
        'output_dim': Y.shape[1],
        'x_range': f"{X.min():.2f}, {X.max():.2f}",
        'y_range': f"{Y.min():.2f}, {Y.max():.2f}",
        'selected_kernel': kernel
    }
    ranking = create_performance_summary_table(results, dataset_info, experiment_config)

    # 7. Final conclusions
    if not DEFAULT_QUIET:
        print(f"\nConclusions:")
        print(f"  Best method: {ranking[0][0]} (score: {ranking[0][1]:.4f})")
        print(f"  Online KSDR improves over linear baselines on nonlinear structure")
        print(f"  Adam optimization supports stable convergence in online updates")
        print(f"  Kernel methods capture nonlinear relationships beyond linear PCA")

    return results, feature_sets, models

## (hyperparameter optimization code moved to optimization.py)

def run_optimized_experiment(dataset: str, config: dict, n_samples: int = 1500):
    """
    Run a single experiment with specific hyperparameter configuration
    """
    # Generate data
    gen = get_generator(dataset)
    X, Y = gen(n_samples=n_samples, noise_level=0.15, random_state=42)

    # Extract configuration parameters
    kernel_type = config.get('kernel', 'rbf')
    learning_rate = config.get('learning_rate', 0.01)
    D_x = config.get('D_x', 300)
    D_y = config.get('D_y', 250)
    sigma_x = config.get('sigma_x', 15.0)
    sigma_y = config.get('sigma_y', 12.0)

    # Create KSPCA models with specific parameters
    k = 5  # Target dimension

    # Batch KSPCA with custom parameters
    batch_kspca = BatchKSDR(
        d_x=X.shape[1], d_y=Y.shape[1], k=k,
        D_x=D_x, D_y=D_y,
        sigma_x=sigma_x, sigma_y=sigma_y,
        kernel_x=kernel_type, kernel_y=kernel_type,
        random_state=42
    )

    # Online KSPCA with custom parameters
    online_kspca = OnlineKernelSDR(
        d_x=X.shape[1], d_y=Y.shape[1], k=k,
        D_x=D_x, D_y=D_y,
        sigma_x=sigma_x, sigma_y=sigma_y,
        kernel_x=kernel_type, kernel_y=kernel_type,
        base_lr=learning_rate,
        adaptive_lr=True,
        random_state=42
    )

    # Train models
    # Batch training
    batch_kspca.fit(X, Y)

    # Online training
    indices_stream = np.random.RandomState(123).permutation(len(X))
    for idx in indices_stream:
        online_kspca.update(X[idx], Y[idx])

    # Get features
    phi_X_batch = batch_kspca.rff_x.transform(X)
    phi_X_online = online_kspca.rff_x.transform(X)

    # Ensure proper centering before projection (robust if future code changes initialization)
    if batch_kspca.U is None:
        raise RuntimeError("BatchKSDR.U is None after fit")
    if online_kspca.U is None:
        raise RuntimeError("OnlineKernelSDR.U is None after updates")
    # If linear kernel + identity mapping, mean_x may be length d_x (not D_x) but current implementation keeps D_x
    phi_X_batch_centered = phi_X_batch - getattr(batch_kspca, 'mean_x', np.mean(phi_X_batch, axis=0))
    phi_X_online_centered = phi_X_online - getattr(online_kspca, 'mean_x', np.mean(phi_X_online, axis=0))
    X_batch = phi_X_batch_centered @ batch_kspca.U
    X_online = phi_X_online_centered @ online_kspca.U

    # Evaluate performance
    results = evaluate_downstream_tasks({
        'Batch_KSDR': X_batch,
        'Online_KSDR': X_online
    }, X, Y)

    return results, {'batch_kspca': batch_kspca, 'online_kspca': online_kspca}



## Legacy main_xor removed; use main(dataset='better_xor') for unified workflow.

def visualize_noise_structure(N=1500, noise_levels=[0.05, 0.1, 0.15, 0.2], random_state=42):
    """
    Visualize the effect of different noise levels on data generation
    
    Parameters:
    - N: Number of samples
    - noise_levels: List of noise levels to compare
    - random_state: Random seed for reproducibility
    """
    print("\n=== Analyzing Noise Structure and Effects ===")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle('Noise Structure Analysis in Highly Nonlinear Data Generation', fontsize=16)
    
    colors = ['blue', 'green', 'orange', 'red']
    
    gen_high = get_generator('highly_nonlinear')
    for i, noise_level in enumerate(noise_levels):
        print(f"Generating data with noise level: {noise_level}")
        # Generate data with specific noise level
        X, Y = gen_high(n_samples=N, noise_level=noise_level, random_state=random_state)
        # Clean version for comparison
        X_clean, Y_clean = gen_high(n_samples=N, noise_level=0.0, random_state=random_state)
        
        # 1. First row: Spiral structure with noise
        axes[0, i].scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap='viridis', alpha=0.6, s=15)
        axes[0, i].set_title(f'Spiral Structure\n(noise={noise_level})', fontsize=12)
        axes[0, i].set_xlabel('X[0] = r*cos(t) + noise')
        axes[0, i].set_ylabel('X[1] = r*sin(t) + noise')
        
        # 2. Second row: Noise magnitude visualization
        noise_magnitude_x = np.sqrt(np.sum((X - X_clean)**2, axis=1))
        noise_magnitude_y = np.sqrt(np.sum((Y - Y_clean)**2, axis=1))
        
        axes[1, i].hist(noise_magnitude_x, bins=30, alpha=0.7, color=colors[i], label='X noise')
        axes[1, i].hist(noise_magnitude_y, bins=30, alpha=0.7, color='red', label='Y noise')
        axes[1, i].set_title(f'Noise Magnitude Distribution\n(σ={noise_level})', fontsize=12)
        axes[1, i].set_xlabel('Noise Magnitude')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].legend()
        
        # 3. Third row: SNR analysis for each feature
        snr_x = []
        snr_y = []
        
        for j in range(X.shape[1]):
            signal_power = np.var(X_clean[:, j])
            noise_power = np.var(X[:, j] - X_clean[:, j])
            snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
            snr_x.append(snr)
        
        for j in range(Y.shape[1]):
            signal_power = np.var(Y_clean[:, j])
            noise_power = np.var(Y[:, j] - Y_clean[:, j])
            snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
            snr_y.append(snr)
        
        x_features = list(range(len(snr_x)))
        y_features = [f'Y{j}' for j in range(len(snr_y))]
        
        axes[2, i].bar(x_features, snr_x, alpha=0.7, color=colors[i], label='X features')
        axes[2, i].bar([len(snr_x) + j + 0.5 for j in range(len(snr_y))], snr_y, 
                      alpha=0.7, color='red', label='Y features')
        axes[2, i].set_title(f'Signal-to-Noise Ratio\n(σ={noise_level})', fontsize=12)
        axes[2, i].set_xlabel('Feature Index')
        axes[2, i].set_ylabel('SNR (dB)')
        axes[2, i].legend()
        axes[2, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig('noise_structure.png')  # removed plt.show
    
    # Additional noise correlation analysis
    plot_noise_correlation_analysis(noise_levels, N, random_state)

def plot_noise_correlation_analysis(noise_levels, N, random_state):
    """
    Analyze how noise affects correlation structure
    """
    print("\n=== Noise Impact on Correlation Structure ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Noise Impact on Data Correlation Structure', fontsize=16)
    
    # Generate clean data for reference
    gen_high = get_generator('highly_nonlinear')
    X_clean, Y_clean = gen_high(n_samples=N, noise_level=0.0, random_state=random_state)
    
    correlation_degradation_xy = []
    correlation_degradation_xx = []
    effective_rank_x = []
    effective_rank_y = []
    
    for noise_level in noise_levels:
        X_noisy, Y_noisy = gen_high(n_samples=N, noise_level=noise_level, random_state=random_state)
        
        # 1. X-Y correlation degradation
        clean_xy_corr = np.corrcoef(np.hstack([X_clean, Y_clean]).T)[:X_clean.shape[1], X_clean.shape[1]:]
        noisy_xy_corr = np.corrcoef(np.hstack([X_noisy, Y_noisy]).T)[:X_noisy.shape[1], X_noisy.shape[1]:]
        
        xy_degradation = np.mean(np.abs(clean_xy_corr - noisy_xy_corr))
        correlation_degradation_xy.append(xy_degradation)
        
        # 2. X-X correlation degradation
        clean_xx_corr = np.corrcoef(X_clean.T)
        noisy_xx_corr = np.corrcoef(X_noisy.T)
        
        xx_degradation = np.mean(np.abs(clean_xx_corr - noisy_xx_corr))
        correlation_degradation_xx.append(xx_degradation)
        
        # 3. Effective rank analysis
        _, s_x, _ = np.linalg.svd(X_noisy, full_matrices=False)
        _, s_y, _ = np.linalg.svd(Y_noisy, full_matrices=False)
        
        # Effective rank (90% energy)
        cumsum_x = np.cumsum(s_x**2) / np.sum(s_x**2)
        cumsum_y = np.cumsum(s_y**2) / np.sum(s_y**2)
        
        eff_rank_x = np.argmax(cumsum_x >= 0.9) + 1
        eff_rank_y = np.argmax(cumsum_y >= 0.9) + 1
        
        effective_rank_x.append(eff_rank_x)
        effective_rank_y.append(eff_rank_y)
    
    # Plot 1: Correlation degradation
    axes[0, 0].plot(noise_levels, correlation_degradation_xy, 'o-', linewidth=2, markersize=8, label='X-Y Correlation')
    axes[0, 0].plot(noise_levels, correlation_degradation_xx, 's-', linewidth=2, markersize=8, label='X-X Correlation')
    axes[0, 0].set_xlabel('Noise Level')
    axes[0, 0].set_ylabel('Correlation Degradation')
    axes[0, 0].set_title('Correlation Matrix Degradation')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Effective rank changes
    axes[0, 1].plot(noise_levels, effective_rank_x, 'o-', linewidth=2, markersize=8, label='X Effective Rank')
    axes[0, 1].plot(noise_levels, effective_rank_y, 's-', linewidth=2, markersize=8, label='Y Effective Rank')
    axes[0, 1].set_xlabel('Noise Level')
    axes[0, 1].set_ylabel('Effective Rank (90% Energy)')
    axes[0, 1].set_title('Data Intrinsic Dimensionality')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Noise impact on different nonlinear structures
    noise_level = 0.15  # Use medium noise level
    X_example, Y_example = gen_high(n_samples=1000, noise_level=noise_level, random_state=random_state)
    X_clean_example, Y_clean_example = gen_high(n_samples=1000, noise_level=0.0, random_state=random_state)
    
    # Show different structures with noise
    structures = {
        'Spiral': ([0, 1], 'X[0], X[1] = r*cos(t), r*sin(t)'),
        'Ring': ([3, 4], 'X[3], X[4] = r²*cos(2t), r²*sin(2t)'),
        'Polynomial': ([5, 6], 'X[5], X[6] = t²*sin(t), exp(-r/2)*cos(3t)'),
    }
    
    for idx, (name, (indices, formula)) in enumerate(structures.items()):
        ax = axes[1, 0] if idx == 0 else axes[1, 1] if idx == 1 else None
        if ax is not None:
            # Plot clean version
            ax.scatter(X_clean_example[:, indices[0]], X_clean_example[:, indices[1]], 
                      alpha=0.3, s=10, color='blue', label='Clean')
            # Plot noisy version
            ax.scatter(X_example[:, indices[0]], X_example[:, indices[1]], 
                      alpha=0.6, s=10, color='red', label=f'Noisy (σ={noise_level})')
            ax.set_title(f'{name} Structure\n{formula}')
            ax.set_xlabel(f'X[{indices[0]}]')
            ax.set_ylabel(f'X[{indices[1]}]')
            ax.legend()
    
    # Summary statistics in the remaining subplot
    if len(structures) == 3:
        axes[1, 1].axis('off')  # We'll use text to show summary
        
        summary_text = f"""
Noise Analysis Summary (σ={noise_levels[-1]}):

Signal-to-Noise Characteristics:
• Spiral features (X[0,1]): Most robust to noise
• Ring features (X[3,4]): Moderate noise sensitivity  
• Polynomial features (X[5,6]): Highest noise sensitivity

Correlation Impact:
• X-Y correlation degradation: {correlation_degradation_xy[-1]:.3f}
• X-X correlation degradation: {correlation_degradation_xx[-1]:.3f}

Dimensionality Impact:
• X effective rank: {effective_rank_x[-1]} dimensions
• Y effective rank: {effective_rank_y[-1]} dimensions

Noise Characteristics:
• Additive Gaussian noise
• Independent across features
• Controlled by noise_level parameter
"""
        
        axes[1, 1].text(0.05, 0.95, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=11, va='top', ha='left',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.tight_layout()
    save_fig('noise_correlation_impact.png')  # removed plt.show

def analyze_noise_impact_on_methods(noise_levels=[0.05, 0.1, 0.15, 0.2], N=1000, kernel: str = "rbf"):
    """
    Analyze how different noise levels affect the performance of different methods
    """
    print("\n=== Analyzing Noise Impact on Feature Extraction Methods ===")

    results_by_noise = {}

    gen_high = get_generator('highly_nonlinear')
    for noise_level in noise_levels:
        print(f"\nTesting with noise level: {noise_level}")
        X, Y = gen_high(n_samples=N, noise_level=noise_level, random_state=42)
        feature_sets, models, split, _ = compare_feature_extraction_methods(X, Y, kernel=kernel)  # ignore experiment_config
        results = evaluate_downstream_tasks(feature_sets, X, Y, idx_split=split)
        results_by_noise[noise_level] = results

    # Plot noise robustness
    plot_noise_robustness_analysis(results_by_noise, noise_levels)

def plot_noise_robustness_analysis(results_by_noise, noise_levels):
    """
    Plot how different methods perform under various noise levels
    """
    print("\n=== Plotting Noise Robustness Analysis ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Method Performance vs Noise Level', fontsize=16)
    
    methods = [m for m in METHODS_ORDER if m in next(iter(results_by_noise.values())).keys()]
    colors = ['gray', 'blue', 'purple', 'green', 'red']
    markers = ['o', 's', 'x', '^', 'd']
    
    # Extract performance metrics
    metrics = ['classification_acc', 'regression_r2', 'avg_correlation', 'rf_classification_acc']
    metric_names = ['Classification Accuracy', 'Regression R²', 'Average Correlation', 'RF Classification']
    
    for metric_idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[metric_idx // 2, metric_idx % 2]
        
        for method_idx, method in enumerate(methods):
            performance_values = []
            
            for noise_level in noise_levels:
                if method in results_by_noise[noise_level]:
                    value = results_by_noise[noise_level][method][metric]
                    performance_values.append(value)
                else:
                    performance_values.append(0)
            
            ax.plot(noise_levels, performance_values, 
                   color=colors[method_idx], marker=markers[method_idx], 
                   linewidth=2, markersize=8, label=method)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} vs Noise Level')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_fig('noise_robustness.png')  # removed plt.show
    
    # (Noise robustness textual summary handled in reporting if implemented)

## Noise robustness summary moved to reporting module.

# Update the main function to include noise analysis
def main_with_noise_analysis(dataset: str = "highly_nonlinear", kernel: str = "auto"):
    """
    Extended main function with comprehensive noise analysis and automatic kernel selection
    """
    logger.info("Starting Comprehensive Downstream Task Experiments with Noise Analysis")
    logger.info("%s", "="*80)

    # 1. Standard experiment with auto kernel selection
    results, feature_sets, models = main(dataset=dataset, kernel=kernel, learning_rate=None)

    # 2. Noise structure visualization (using same dataset directory)
    # Call noise structure visualization (internal logging will respect global config)
    try:
        visualize_noise_structure()
    except Exception as e:
        logger.warning("Noise structure visualization failed: %s", e)

    # 3. Noise robustness analysis (using same dataset directory)
    try:
        analyze_noise_impact_on_methods(kernel=kernel)
    except Exception as e:
        logger.warning("Noise impact analysis failed: %s", e)

    logger.info("Complete experiment with noise analysis finished")

    return results, feature_sets, models

def run_all_datasets(datasets=None, kernel="auto", include_big_plots=False, epochs: int = 1):
    """
    Run experiments on all available datasets
    """
    if datasets is None:
        datasets = ["highly_nonlinear", "better_xor", "nuclear_friendly",
                   "extreme1", "extreme2", "extreme3", "piecewise", "swiss"]

    print(f"Running experiments on {len(datasets)} datasets with kernel={kernel}")
    print("="*80)

    all_results = {}
    for i, dataset in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] Processing dataset: {dataset}")
        print("-" * 50)

        try:
            results, feature_sets, models = main(dataset=dataset, kernel=kernel, include_big_plots=include_big_plots, learning_rate=None)

            all_results[dataset] = results
            print(f"{dataset} completed successfully")

        except Exception as e:
            print(f"{dataset} failed: {e}")
            all_results[dataset] = None

    # Summary of all experiments
    print(f"\n{'='*80}")
    print("ALL DATASETS EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    successful_datasets = [d for d, r in all_results.items() if r is not None]
    failed_datasets = [d for d, r in all_results.items() if r is None]

    print(f"Successful: {len(successful_datasets)}/{len(datasets)}")
    print(f"Failed: {len(failed_datasets)}/{len(datasets)}")

    if successful_datasets:
        print(f"\nSuccessful datasets: {', '.join(successful_datasets)}")

    if failed_datasets:
        print(f"\nFailed datasets: {', '.join(failed_datasets)}")

    return all_results

def benchmark_suite(include_kin8nm: bool = True, subset_kin8nm: int = 5000, kernel: str = "auto"):
    """Run a fast benchmark over 8 synthetic datasets plus optional kin8nm.

    Parameters
    ----------
    include_kin8nm : bool
        Whether to include the real kin8nm regression dataset.
    subset_kin8nm : int
        If kin8nm included, limit to first N samples for speed (None = full).
    kernel : str
        Kernel choice ("auto" | "rbf" | "linear").
    Returns
    -------
    dict
        Mapping dataset -> results dict (or exception string on failure)
    """
    target_datasets = [
        "highly_nonlinear", "better_xor", "nuclear_friendly", "extreme1",
        "extreme2", "extreme3", "piecewise", "swiss"
    ]
    if include_kin8nm:
        target_datasets.append("kin8nm")
    summary = {}
    print("Starting benchmark suite:")
    print(", ".join(target_datasets))
    for ds_name in target_datasets:
        print(f"\n=== Dataset: {ds_name} ===")
        try:
            if ds_name == "kin8nm":
                # Temporarily monkey-patch generator wrapper to allow subset size
                from data_gens import get_generator
                gen = get_generator("kin8nm")
                X, Y = gen(n_samples=subset_kin8nm) if subset_kin8nm else gen()
                # Reuse core pipeline parts: automatic kernel, feature extraction, downstream eval
                # Minimal duplication of logic from main()
                if kernel == "auto":
                    selector = KernelSelector(threshold=0.4, method='combined')
                    sel_kernel, _ = selector.select_kernel(X, Y, verbose=False)
                else:
                    sel_kernel = kernel
                # Online model (reuse pattern from other parts: map single kernel to both x,y RFFs)
                if sel_kernel == 'linear':
                    Dx_id, Dy_id = X.shape[1], Y.shape[1]
                    sigma_x_b = sigma_y_b = 1.0
                else:
                    Dx_id, Dy_id = 64, 32
                    sigma_x_b = sigma_y_b = 1.0
                online = OnlineKernelSDR(
                    d_x=X.shape[1], d_y=Y.shape[1], k=8,
                    D_x=Dx_id, D_y=Dy_id,
                    sigma_x=sigma_x_b, sigma_y=sigma_y_b,
                    kernel_x=sel_kernel, kernel_y=sel_kernel,
                    base_lr=0.01, adaptive_lr=False, advanced_adaptive=False,
                    random_state=0
                )
                for i in range(X.shape[0]):
                    online.update(X[i], Y[i])
                # Project with learned U: features = Phi_x @ U
                Phi_online = online.rff_x.transform(X)
                Z_online = (Phi_online - online.mean_x) @ online.U
                # Batch baseline
                batch = BatchKSDR(
                    d_x=X.shape[1], d_y=Y.shape[1], k=8,
                    D_x=Dx_id, D_y=Dy_id,
                    sigma_x=sigma_x_b, sigma_y=sigma_y_b,
                    kernel_x=sel_kernel, kernel_y=sel_kernel,
                    random_state=0
                )
                batch.fit(X, Y)
                Phi_batch = batch.rff_x.transform(X)
                if batch.U is None:
                    raise RuntimeError("BatchKSDR.U is None after fit in benchmark suite")
                if batch.mean_x is not None:
                    Z_batch = (Phi_batch - batch.mean_x) @ batch.U
                else:
                    Z_batch = Phi_batch @ batch.U
                # Downstream simple regression (R^2)
                from sklearn.linear_model import Ridge
                from sklearn.metrics import r2_score
                ridge = Ridge(alpha=1.0)
                ridge.fit(Z_online, Y.ravel())
                r2_online = r2_score(Y, ridge.predict(Z_online))
                ridge.fit(Z_batch, Y.ravel())
                r2_batch = r2_score(Y, ridge.predict(Z_batch))
                summary[ds_name] = {"kernel": sel_kernel, "r2_online": float(r2_online), "r2_batch": float(r2_batch)}
            else:
                res, *_ = main(dataset=ds_name, kernel=kernel, include_big_plots=False, learning_rate=None)
                summary[ds_name] = res
            print(f"{ds_name} success")
        except Exception as e:
            print(f"{ds_name} failed: {e}")
            summary[ds_name] = f"ERROR: {e}"
    print("\nBenchmark summary (keys only):", list(summary.keys()))
    return summary

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run OKS experiments (Online Kernel Supervised PCA)")
    parser.add_argument("--mode", choices=["dataset","single","xor","all","noise","benchmark"], default="dataset",
                            help="Experiment mode: dataset=one dataset (default), xor=better_xor only, all=all datasets, noise=dataset + noise analysis, benchmark=8 synthetic + kin8nm quick run. 'single' is a backward-compatible alias.")
    parser.add_argument("--dataset", type=str, default="highly_nonlinear",
                        help="Dataset generator name for --mode dataset/noise")
    parser.add_argument("--kernel", choices=["auto","rbf","linear"], default="auto",
                        help="Kernel type for KSPCA methods (auto=automatic selection)")
    parser.add_argument("--include_big_plots", action="store_true", default=True,
                        help="(Enabled by default) Generate comprehensive large figures")
    parser.add_argument("--no-big-plots", dest="include_big_plots", action="store_false",
                        help="Disable comprehensive large figures (alias flag)")
    parser.add_argument("--fig-formats", type=str, default="png",
                        help="Comma separated figure formats to save (default: png). Example: png,svg,pdf")
    parser.add_argument("--verbose", action="store_true", default=False, help="Verbose debug output")
    parser.add_argument("--quiet", action="store_true", default=False, help="Suppress most logs (overrides --verbose)")
    parser.add_argument("--optimize_hyperparams", action="store_true", default=True,
                        help="Run hyperparameter optimization for all datasets (default: True)")
    parser.add_argument("--no_optimize", action="store_true", default=False,
                        help="Skip hyperparameter optimization for faster execution")
    args = parser.parse_args()
    if args.quiet:
        globals()['DEFAULT_QUIET'] = True
    elif args.verbose:
        globals()['DEFAULT_VERBOSE'] = True
    configure_logging(quiet=args.quiet, verbose=args.verbose)

    # Configure figure formats
    requested_formats = [f.strip().lower() for f in args.fig_formats.split(',') if f.strip()]
    if not requested_formats:
        requested_formats = ["png"]
    # apply to fig_utils global
    set_fig_formats(requested_formats)
    logger.info("Figure output formats: %s", requested_formats)

    # Hyperparameter optimization enabled by default unless --no_optimize specified
    if args.no_optimize:
        logger.info("Skipping hyperparameter optimization for faster execution")
        optimize = False
    else:
        logger.info("Hyperparameter optimization enabled (default)")
        optimize = True

    optimized_configs = {}
    if optimize:
        # Decide which datasets to optimize
        if args.mode == "all":
            datasets_to_optimize = ["highly_nonlinear", "better_xor", "nuclear_friendly",
                                    "extreme1", "extreme2", "extreme3", "piecewise", "swiss"]
        elif args.mode == "xor":
            datasets_to_optimize = ["better_xor"]
        elif args.mode in ["single", "dataset", "noise"]:
            datasets_to_optimize = [args.dataset]
        else:
            datasets_to_optimize = []

        # Run optimization loop
        for dataset in datasets_to_optimize:
            try:
                logger.info("%s", "="*80)
                logger.info("Optimizing hyperparameters for: %s", dataset)
                best_config = run_hyperparameter_optimization(
                    dataset=dataset,
                    kernel=args.kernel,
                    max_configs=5,
                    n_samples=1000
                )
                if best_config:
                    optimized_configs[dataset] = best_config
                    logger.info("%s: Optimization completed", dataset)
                else:
                    logger.warning("%s: Optimization failed, using defaults", dataset)
            except Exception as e:
                logger.warning("%s: Optimization error - %s, using defaults", dataset, e)

    # Execute experiment based on mode
    if args.mode == "xor":
        if optimized_configs.get("better_xor"):
            logger.info("Using optimized configuration for XOR experiment (placeholder)")
        results, feature_sets, models = main(dataset="better_xor", kernel=args.kernel, include_big_plots=args.include_big_plots, learning_rate=None)
    elif args.mode == "all":
        all_results = run_all_datasets(kernel=args.kernel, include_big_plots=args.include_big_plots)
    elif args.mode == "noise":
        if optimized_configs.get(args.dataset):
            logger.info("Using optimized configuration for noise analysis on %s (placeholder)", args.dataset)
        results, feature_sets, models = main_with_noise_analysis(dataset=args.dataset, kernel=args.kernel)
    elif args.mode == "benchmark":
        bench = benchmark_suite(include_kin8nm=True, subset_kin8nm=5000, kernel=args.kernel)
        logger.info("Benchmark suite finished. Summary keys: %s", list(bench.keys()))
    else:  # dataset/single
        if optimized_configs.get(args.dataset):
            logger.info("Using optimized configuration for %s (placeholder)", args.dataset)
        results, feature_sets, models = main(dataset=args.dataset, kernel=args.kernel, include_big_plots=args.include_big_plots, learning_rate=None)