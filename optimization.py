"""Hyperparameter optimization utilities extracted from ds.py.

Functions:
- get_adaptive_configurations(dataset)
- run_optimized_experiment(dataset, config, n_samples=1500)
- calculate_optimization_score(metrics, method_name="Online_KSPCA")
- run_hyperparameter_optimization(...)

The goal is to keep ds.py lean; all optimization related code lives here.
"""
from __future__ import annotations
import time
import numpy as np
from data_gens import get_generator
from mainFunction.OKS_main import OnlineKernelSDR
from mainFunction.OKS_batch import BatchKSDR
from kernel_selector import KernelSelector

# ---- Scoring ----

def calculate_optimization_score(metrics, method_name: str = "Online_KSDR") -> float:
    if method_name not in metrics:
        return 0.0
    m = metrics[method_name]
    classification_weight = 0.4
    regression_weight = 0.4
    correlation_weight = 0.2
    score = (classification_weight * m['classification_acc'] +
             regression_weight * max(0, m['regression_r2']) +
             correlation_weight * m['avg_correlation'])
    dim_penalty = 1.0 / (1.0 + m['feature_dim'] / 10.0)
    return score * dim_penalty

# ---- Configuration space ----

def get_adaptive_configurations(dataset: str):
    if dataset in ['extreme1', 'extreme2', 'extreme3']:
        return [
            {'kernel': 'rbf', 'learning_rate': 0.005, 'D_x': 400, 'D_y': 300, 'sigma_x': 10.0, 'sigma_y': 10.0},
            {'kernel': 'rbf', 'learning_rate': 0.01, 'D_x': 600, 'D_y': 450, 'sigma_x': 15.0, 'sigma_y': 12.0},
            {'kernel': 'rbf', 'learning_rate': 0.002, 'D_x': 800, 'D_y': 600, 'sigma_x': 8.0, 'sigma_y': 8.0},
        ]
    elif dataset in ['swiss', 'piecewise']:
        return [
            {'kernel': 'rbf', 'learning_rate': 0.002, 'D_x': 300, 'D_y': 250, 'sigma_x': 12.0, 'sigma_y': 10.0},
            {'kernel': 'rbf', 'learning_rate': 0.005, 'D_x': 500, 'D_y': 400, 'sigma_x': 18.0, 'sigma_y': 15.0},
            {'kernel': 'linear', 'learning_rate': 0.01, 'D_x': 200, 'D_y': 150},
        ]
    elif dataset in ['better_xor', 'nuclear_friendly']:
        return [
            {'kernel': 'rbf', 'learning_rate': 0.01, 'D_x': 200, 'D_y': 150, 'sigma_x': 5.0, 'sigma_y': 5.0},
            {'kernel': 'rbf', 'learning_rate': 0.02, 'D_x': 300, 'D_y': 200, 'sigma_x': 8.0, 'sigma_y': 6.0},
            {'kernel': 'rbf', 'learning_rate': 0.005, 'D_x': 400, 'D_y': 250, 'sigma_x': 12.0, 'sigma_y': 10.0},
        ]
    else:
        return [
            {'kernel': 'rbf', 'learning_rate': 0.01, 'D_x': 300, 'D_y': 250, 'sigma_x': 15.0, 'sigma_y': 12.0},
            {'kernel': 'rbf', 'learning_rate': 0.005, 'D_x': 500, 'D_y': 400, 'sigma_x': 20.0, 'sigma_y': 15.0},
            {'kernel': 'rbf', 'learning_rate': 0.02, 'D_x': 200, 'D_y': 150, 'sigma_x': 10.0, 'sigma_y': 10.0},
            {'kernel': 'linear', 'learning_rate': 0.01, 'D_x': 150, 'D_y': 120},
        ]

# ---- Single configuration run ----

def run_optimized_experiment(dataset: str, config: dict, n_samples: int = 1500):
    gen = get_generator(dataset)
    X, Y = gen(n_samples=n_samples, noise_level=0.15, random_state=42)
    k = 5
    kernel_type = config.get('kernel', 'rbf')
    lr = config.get('learning_rate', 0.01)
    # For linear kernel we enforce identity mapping: D_x/d_x and D_y/d_y
    if kernel_type == 'linear':
        D_x = X.shape[1]
        D_y = Y.shape[1]
    else:
        D_x = config.get('D_x', 300)
        D_y = config.get('D_y', 250)
    sigma_x = config.get('sigma_x', 15.0)
    sigma_y = config.get('sigma_y', 12.0)
    batch_kspca = BatchKSDR(d_x=X.shape[1], d_y=Y.shape[1], k=k,
                             D_x=D_x, D_y=D_y,
                             sigma_x=sigma_x, sigma_y=sigma_y,
                             kernel_x=kernel_type, kernel_y=kernel_type,
                             random_state=42)
    online_kspca = OnlineKernelSDR(d_x=X.shape[1], d_y=Y.shape[1], k=k,
                                   D_x=D_x, D_y=D_y,
                                   sigma_x=sigma_x, sigma_y=sigma_y,
                                   kernel_x=kernel_type, kernel_y=kernel_type,
                                   base_lr=lr, adaptive_lr=True,
                                   random_state=42)
    batch_kspca.fit(X, Y)
    for idx in np.random.RandomState(123).permutation(len(X)):
        online_kspca.update(X[idx], Y[idx])
    phi_X_batch = batch_kspca.rff_x.transform(X)
    phi_X_online = online_kspca.rff_x.transform(X)
    if batch_kspca.U is None or online_kspca.U is None:
        raise RuntimeError("Projection matrices not learned")
    X_batch = phi_X_batch @ batch_kspca.U
    X_online = phi_X_online @ online_kspca.U
    from ds import evaluate_downstream_tasks  # local import to avoid circular heavy imports at module load
    results = evaluate_downstream_tasks({'Batch_KSDR': X_batch,'Online_KSDR': X_online}, X, Y)
    return results, {'batch_kspca': batch_kspca, 'online_kspca': online_kspca}

# ---- Main optimization routine ----

def run_hyperparameter_optimization(dataset: str = "better_xor", kernel: str = "auto",
                                    max_configs: int = 5, random_seed: int = 42,
                                    n_samples: int = 1500):
    print(f"Starting Enhanced Hyperparameter Optimization for {dataset}")
    print("="*80)
    print(f"Dataset: {dataset}")
    print(f"Target: Optimize Online KSDR performance")
    print(f"Max configs to test: {max_configs}")
    print(f"Sample size: {n_samples}")
    print(f"Kernel type: {kernel}")
    if kernel == "auto":
        print(f"Auto kernel selection for {dataset}...")
        X_sample, Y_sample = get_generator(dataset)(n_samples=1000, noise_level=0.15, random_state=42)
        selector = KernelSelector(threshold=0.4, method='combined')
        selected_kernel, _ = selector.select_kernel(X_sample, Y_sample, verbose=True)
        print(f"KernelSelector chose: {selected_kernel}")
        kernel = selected_kernel
    else:
        print(f"Using specified kernel: {kernel}")

    # --- Linear kernel short-circuit: identity mapping, no real hyperparameters ---
    if kernel == 'linear':
        print("Linear kernel detected -> skipping hyperparameter search (identity mapping).")
        # Sample once to know dimensions
        X_tmp, Y_tmp = get_generator(dataset)(n_samples= min(200, n_samples), noise_level=0.15, random_state=999)
        identity_config = {'kernel': 'linear', 'learning_rate': 0.0, 'D_x': X_tmp.shape[1], 'D_y': Y_tmp.shape[1]}
        try:
            results, models = run_optimized_experiment(dataset, identity_config, n_samples)
            score_online = calculate_optimization_score(results, 'Online_KSDR')
            score_batch = calculate_optimization_score(results, 'Batch_KSDR')
            print("Optimization (trivial) complete.")
            print("Online score:", f"{score_online:.4f}")
            print("Batch  score:", f"{score_batch:.4f}")
            return {
                'best_config': identity_config,
                'best_online_score': score_online,
                'best_batch_score': score_batch,
                'ranking': [{
                    'config': 'identity',
                    'online_score': score_online,
                    'batch_score': score_batch,
                    'time': 0.0,
                    'params': identity_config
                }],
                'total_time': 0.0,
                'successful_configs': 1,
                'total_configs': 1,
                'dataset': dataset,
                'kernel': 'linear'
            }
        except Exception as e:
            print(f"Linear identity evaluation failed: {e}")
            return {
                'best_config': None,
                'ranking': [],
                'total_time': 0.0,
                'successful_configs': 0,
                'total_configs': 1,
                'dataset': dataset,
                'kernel': 'linear',
                'error': str(e)
            }
    all_configs = get_adaptive_configurations(dataset)
    rng = np.random.default_rng(random_seed)
    if len(all_configs) > max_configs:
        idx_sel = rng.choice(len(all_configs), size=max_configs, replace=False)
        configs = [all_configs[i] for i in idx_sel]
        print(f"Randomly selected {max_configs} out of {len(all_configs)} configurations")
    else:
        configs = all_configs
    for c in configs:
        c['kernel'] = kernel
    print(f"Optimizing hyperparameters for {dataset} with {kernel} kernel")
    print(f"Testing {len(configs)} configurations:")
    for i,c in enumerate(configs,1):
        print(f"   {i}. {c}")
    results_summary = []
    best_results = {}
    total_start = time.time()
    for i,c in enumerate(configs,1):
        cname = f"config_{i}"
        print(f"\nTesting configuration {i}/{len(configs)}: {cname}")
        print(f"   Parameters: {c}")
        try:
            start = time.time()
            results, models = run_optimized_experiment(dataset, c, n_samples)
            elapsed = time.time() - start
            online_score = calculate_optimization_score(results, "Online_KSDR")
            batch_score = calculate_optimization_score(results, "Batch_KSDR")
            entry = {
                'config': c.copy(),
                'online_score': online_score,
                'batch_score': batch_score,
                'results': results,
                'models': models,
                'elapsed': elapsed,
                'name': cname
            }
            results_summary.append(entry)
            print(f"   Success. Time: {elapsed:.1f}s")
            print(f"   Online KSDR Score: {online_score:.4f}")
            print(f"   Batch KSDR1 Score: {batch_score:.4f}")
        except Exception as e:
            print(f"   [Fail] {e}")
    if not results_summary:
        print("No successful configurations.")
        return {}
    results_summary.sort(key=lambda x: x['online_score'], reverse=True)
    best = results_summary[0]
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print(f"Best Configuration: {best['name']}")
    print("Performance:")
    print(f"   Online KSDR Score: {best['online_score']:.4f}")
    print(f"   Batch KSDR Score: {best['batch_score']:.4f}")
    print(f"   Best Time: {best['elapsed']:.2f}s")
    print(f"Total optimization time: {time.time()-total_start:.1f}s")
    for k,v in best['config'].items():
        print(f"   {k}: {v}")
    print("\nConfiguration Ranking:")
    print("Rank Config     Online   Batch    Time")
    print("-"*50)
    for rank,e in enumerate(results_summary,1):
        print(f"{rank:<4} {e['name']:<9} {e['online_score']:<8.4f} {e['batch_score']:<8.4f} {e['elapsed']:<5.1f}")
    best_results = {
        'best_config': best['config'],
        'best_online_score': best['online_score'],
        'best_batch_score': best['batch_score'],
        'ranking': results_summary,
    }
    return best_results

__all__ = [
    'get_adaptive_configurations',
    'run_optimized_experiment',
    'run_hyperparameter_optimization',
    'calculate_optimization_score'
]
