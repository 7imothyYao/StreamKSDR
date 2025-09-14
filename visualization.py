"""Visualization utilities extracted from ds.py for cleaner separation.

Functions here should remain side-effect free except for file outputs via fig_utils.save_fig.
They respect quiet mode implicitly (avoid printing unless essential)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from fig_utils import save_fig

def create_small_format_visualizations(X, Y, feature_sets, models, results):
    """Create small format visualizations suitable for academic papers."""
    create_performance_comparison_chart(results)
    create_feature_embedding_visualization(X, Y, feature_sets, Y)
    create_correlation_heatmap(feature_sets, Y)

def create_performance_comparison_chart(results):
    methods = list(results.keys())
    metrics = ['classification_acc', 'regression_r2', 'avg_correlation']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = [results[m][metric] for m in methods]
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bars = ax.bar(methods, values, color=colors[:len(methods)], alpha=0.8)
        for bar, value in zip(bars, values):
            h = bar.get_height(); ax.text(bar.get_x()+bar.get_width()/2., h + (0.01 if h>=0 else -0.03), f'{value:.3f}', ha='center', va='bottom' if h>=0 else 'top', fontsize=9)
        ax.set_title(metric.replace('_',' ').title(), fontsize=12)
        ax.set_ylabel(metric.replace('_',' ').title(), fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        if metric == 'regression_r2':
            min_val, max_val = min(values), max(values)
            margin = (max_val - min_val) * 0.1 if max_val>min_val else 0.1
            ax.set_ylim(min(min_val - margin, -0.2), max(max_val + margin, 0.8))
        else:
            max_val = max(values)
            upper = min(max_val * 1.1, 1.0) if metric == 'classification_acc' else max_val * 1.15
            if upper < 0.05: upper = 0.05
            ax.set_ylim(0, upper)
    plt.tight_layout(); save_fig('performance_comparison.png'); plt.close()

def create_feature_embedding_visualization(X, Y, feature_sets, Y_full):
    methods = [m for m in ['Raw','PCA','Online_KSPCA'] if m in feature_sets]
    if not methods: return
    fig, axes = plt.subplots(1, len(methods), figsize=(4*len(methods),4))
    if len(methods)==1: axes = [axes]
    for ax, method in zip(axes, methods):
        feats = feature_sets.get(method)
        if feats is None: continue
        if feats.shape[1] > 2:
            from sklearn.decomposition import PCA as _PCA
            feats_2d = _PCA(n_components=2, random_state=42).fit_transform(feats)
        else:
            feats_2d = feats
        sc = ax.scatter(feats_2d[:,0], feats_2d[:,1], c=Y_full[:,0], cmap='viridis', s=16, alpha=0.65)
        ax.set_title(f'{method} (2D proj)', fontsize=11)
        ax.set_xlabel('Comp1', fontsize=9); ax.set_ylabel('Comp2', fontsize=9)
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8); cbar.ax.tick_params(labelsize=8)
    plt.tight_layout(w_pad=1.0); save_fig('feature_embeddings.png'); plt.close()

def create_correlation_heatmap(feature_sets, Y):
    methods = [m for m in ['Raw','PCA','Online_KSPCA'] if m in feature_sets]
    corrs = {}
    for m in methods:
        feats = feature_sets.get(m)
        if feats is None: continue
        C = np.zeros((feats.shape[1], Y.shape[1]))
        for i in range(feats.shape[1]):
            for j in range(Y.shape[1]):
                try:
                    C[i,j] = abs(np.corrcoef(feats[:,i], Y[:,j])[0,1])
                except Exception:
                    C[i,j] = 0.0
        corrs[m] = C
    if not methods: return
    fig, axes = plt.subplots(1, len(methods), figsize=(4.5*len(methods),4.2))
    if len(methods)==1: axes=[axes]
    for ax, m in zip(axes, methods):
        data = corrs.get(m)
        if data is None: continue
        nf = min(10, data.shape[0]); nt = min(5, data.shape[1])
        im = ax.imshow(data[:nf,:nt], cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(nt)); ax.set_xticklabels([f'Y{j}' for j in range(nt)], fontsize=7)
        ax.set_yticks(range(nf)); ax.set_yticklabels([f'F{j}' for j in range(nf)], fontsize=7)
        ax.set_title(f'{m} Corr', fontsize=11)
        cbar = plt.colorbar(im, ax=ax, shrink=0.75, pad=0.02); cbar.ax.tick_params(labelsize=7)
    plt.tight_layout(w_pad=1.0); save_fig('correlation_heatmap.png'); plt.close()
