"""Reporting & extended analysis utilities extracted from ds.py.

Contains: comprehensive visualization figure, learning curves, noise structure/robustness analysis.
All plotting uses non-interactive backend (Agg). Functions avoid printing unless informative.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from fig_utils import save_fig, set_dataset_save_dir
import fig_utils
from mainFunction.OKS_main import OnlineKernelSDR
from mainFunction.OKS_batch import BatchKSDR

def create_comprehensive_visualizations(X, Y, feature_sets, models, results, methods_order=None, quiet=False):
    if quiet:
        # Skip verbose logging
        pass
    else:
        print("\n=== Generating Visualization Results ===")
    if methods_order is None:
        methods_order = ['Raw','PCA','Batch_KSDR','Online_KSDR']
    plt.rcParams['figure.figsize'] = (15, 12)
    fig = plt.figure(figsize=(20,15))
    use_high_dim_viz = X.shape[1] > 50 or Y.shape[1] > 10
    # (Simplified: import minimal parts from original for correlation + performance summary panels)
    # Original correlation + cross-modal
    if use_high_dim_viz:
        plt.subplot(3,4,1)
        corr_X = np.corrcoef(X.T)
        mask = np.triu(np.ones_like(corr_X, dtype=bool), k=1)
        corr_values = corr_X[mask]
        plt.hist(corr_values, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(np.mean(corr_values), color='red', linestyle='--', label=f'Mean: {np.mean(corr_values):.3f}')
        plt.axvline(np.median(corr_values), color='orange', linestyle='--', label=f'Median: {np.median(corr_values):.3f}')
        plt.title(f'X Correlation Dist (n={X.shape[1]})'); plt.legend(); plt.grid(True, alpha=0.3)
        plt.subplot(3,4,2)
        cross_corr = np.corrcoef(np.hstack([X,Y]).T)[:X.shape[1], X.shape[1]:]
        max_corr_per_y = np.max(np.abs(cross_corr), axis=0)
        mean_corr_per_y= np.mean(np.abs(cross_corr), axis=0)
        y_idx = np.arange(len(max_corr_per_y))
        plt.bar(y_idx, max_corr_per_y, alpha=0.7, color='lightcoral', label='Max |corr|')
        plt.bar(y_idx, mean_corr_per_y, alpha=0.7, color='lightblue', label='Mean |corr|')
        plt.title('X-Y Cross-Modal Corr'); plt.legend(); plt.grid(True, alpha=0.3)
    else:
        plt.subplot(3,4,1)
        corr_X = np.corrcoef(X.T)
        sns.heatmap(corr_X, annot=True, cmap='coolwarm', center=0, square=True, fmt='.2f', cbar_kws={'shrink':0.8})
        plt.title('X Corr Matrix')
        plt.subplot(3,4,2)
        cross_corr = np.corrcoef(np.hstack([X,Y]).T)[:X.shape[1], X.shape[1]:]
        sns.heatmap(cross_corr, annot=True, cmap='viridis', square=False, fmt='.2f', cbar_kws={'shrink':0.8})
        plt.title('X-Y Corr Matrix')
    # Y distributions / variance features
    plt.subplot(3,4,3)
    for i in range(min(3, Y.shape[1])):
        plt.hist(Y[:,i], alpha=0.7, bins=30, label=f'Y[{i}]')
    plt.legend(); plt.title('Y Distributions')
    # Feature spaces (first two methods only to limit size)
    feat_methods = [m for m in methods_order if m in feature_sets][:4]
    for i, name in enumerate(feat_methods):
        plt.subplot(3,4,5+i)
        feats = feature_sets[name]
        if feats.shape[1] >= 2:
            sc = plt.scatter(feats[:,0], feats[:,1], c=Y[:,0], cmap='viridis', alpha=0.6, s=20)
            plt.colorbar(sc, shrink=0.8)
        else:
            plt.hist(feats[:,0], bins=30, alpha=0.7)
        plt.title(f'{name} Space')
    plt.tight_layout()
    save_fig('overview_feature_performance.png')
    plt.close(fig)

def plot_online_learning_curves(X, Y, online_model, quiet=False):
    if not quiet:
        print("\n=== Plotting Online Learning Curves ===")
    N = X.shape[0]
    checkpoints = np.unique(np.logspace(1, np.log10(N), 20).astype(int))
    d_x, d_y = X.shape[1], Y.shape[1]
    k_latent = getattr(online_model,'k',3)
    D_x = getattr(online_model,'D_x',150); D_y = getattr(online_model,'D_y',120)
    sigma_x = getattr(online_model,'sigma_x',15.0); sigma_y = getattr(online_model,'sigma_y',15.0)
    base_lr = getattr(online_model,'base_lr',0.01)
    learner = OnlineKernelSDR(d_x=d_x,d_y=d_y,k=k_latent,D_x=D_x,D_y=D_y,sigma_x=sigma_x,sigma_y=sigma_y,kernel_x=getattr(online_model,'kernel_x','rbf'),kernel_y=getattr(online_model,'kernel_y','rbf'),base_lr=base_lr,adaptive_lr=True,random_state=42)
    batch = BatchKSDR(d_x,d_y,k=k_latent,D_x=D_x,D_y=D_y,sigma_x=sigma_x,sigma_y=sigma_y,kernel_x=getattr(online_model,'kernel_x','rbf'),kernel_y=getattr(online_model,'kernel_y','rbf'),random_state=42)
    batch.fit(X,Y)
    order = np.random.RandomState(123).permutation(N)
    errors, angles, corrs, samples = [],[],[],[]
    idx_ck = 0
    for step,i in enumerate(order,1):
        learner.update(X[i],Y[i])
        if step == checkpoints[idx_ck]:
            if batch.U is not None and learner.U is not None:
                Pb = batch.U @ batch.U.T; Po = learner.U @ learner.U.T
                proj_err = np.linalg.norm(Pb-Po,'fro')/np.linalg.norm(Pb,'fro')
                Ub,_ = np.linalg.qr(batch.U); Uo,_ = np.linalg.qr(learner.U)
                svals = np.linalg.svd(Ub.T@Uo, compute_uv=False)
                errors.append(proj_err*100); angles.append(np.degrees(np.arccos(np.clip(svals[0],0,1)))); corrs.append(svals[0]); samples.append(step)
            idx_ck += 1
            if idx_ck >= len(checkpoints): break
    fig, axes = plt.subplots(2,2, figsize=(15,10))
    axes[0,0].semilogx(samples, errors, 'b-o'); axes[0,0].set_title('Projection Error (%)')
    axes[0,1].semilogx(samples, angles, 'r-s'); axes[0,1].set_title('Principal Angle')
    axes[1,0].semilogx(samples, corrs, 'g-^'); axes[1,0].set_title('Canonical Corr')
    axes[1,1].axis('off'); axes[1,1].text(0.05,0.9,f'Final Error: {errors[-1]:.2f}%',fontsize=12)
    save_fig('online_learning_curves.png'); plt.close(fig)

def visualize_noise_structure(N=1500, noise_levels=None, random_state=42, quiet=False):
    if noise_levels is None: noise_levels=[0.05,0.1,0.15,0.2]
    if not quiet:
        print("\n=== Analyzing Noise Structure and Effects ===")
    from data_gens import get_generator
    gen_high = get_generator('highly_nonlinear')
    fig, axes = plt.subplots(3, len(noise_levels), figsize=(5*len(noise_levels),15))
    colors = ['blue','green','orange','red']
    for i,nl in enumerate(noise_levels):
        X,Y = gen_high(n_samples=N, noise_level=nl, random_state=random_state)
        Xc,Yc = gen_high(n_samples=N, noise_level=0.0, random_state=random_state)
        axes[0,i].scatter(X[:,0],X[:,1], c=Y[:,0], cmap='viridis', s=15, alpha=0.6)
        axes[0,i].set_title(f'nois={nl}')
        nm_x = np.sqrt(np.sum((X-Xc)**2, axis=1)); nm_y = np.sqrt(np.sum((Y-Yc)**2, axis=1))
        axes[1,i].hist(nm_x, bins=30, alpha=0.7, color=colors[i]); axes[1,i].hist(nm_y, bins=30, alpha=0.5, color='red')
        axes[1,i].set_title('Noise Magnitude')
        snr_x=[]; snr_y=[]
        for j in range(X.shape[1]):
            sp = np.var(Xc[:,j]); npow=np.var(X[:,j]-Xc[:,j]); snr_x.append(10*np.log10(sp/(npow+1e-8)))
        for j in range(Y.shape[1]):
            sp = np.var(Yc[:,j]); npow=np.var(Y[:,j]-Yc[:,j]); snr_y.append(10*np.log10(sp/(npow+1e-8)))
        axes[2,i].bar(range(len(snr_x)), snr_x, alpha=0.7, color=colors[i])
        axes[2,i].bar([len(snr_x)+j+0.5 for j in range(len(snr_y))], snr_y, alpha=0.7, color='red')
    plt.tight_layout(); save_fig('noise_structure.png'); plt.close(fig)

def analyze_noise_impact_on_methods(noise_levels=None, N=1000, kernel='rbf', quiet=False):
    if noise_levels is None: noise_levels=[0.05,0.1,0.15,0.2]
    if not quiet:
        print("\n=== Analyzing Noise Impact on Methods ===")
    from data_gens import get_generator
    from ds import compare_feature_extraction_methods, evaluate_downstream_tasks
    gen_high = get_generator('highly_nonlinear')
    results_by_noise = {}
    for nl in noise_levels:
        X,Y = gen_high(n_samples=N, noise_level=nl, random_state=42)
        feats, models, split, _ = compare_feature_extraction_methods(X,Y,kernel=kernel)
        res = evaluate_downstream_tasks(feats,X,Y,idx_split=split)
        results_by_noise[nl]=res
    plot_noise_robustness_analysis(results_by_noise, noise_levels)

def plot_noise_robustness_analysis(results_by_noise, noise_levels):
    methods = list(next(iter(results_by_noise.values())).keys())
    fig, axes = plt.subplots(2,2, figsize=(15,12)); metrics=['classification_acc','regression_r2','avg_correlation','rf_classification_acc']; names=['Cls Acc','Reg R2','Avg Corr','RF Acc']
    colors=['gray','blue','purple','green','red']; markers=['o','s','x','^','d']
    for idx,(mname,label) in enumerate(zip(metrics,names)):
        ax=axes[idx//2][idx%2]
        for mi,method in enumerate(methods):
            vals=[results_by_noise[nl][method][mname] for nl in noise_levels]
            ax.plot(noise_levels, vals, color=colors[mi%len(colors)], marker=markers[mi%len(markers)], label=method)
        ax.set_title(f'{label} vs Noise'); ax.set_xlabel('Noise'); ax.set_ylabel(label); ax.grid(True, alpha=0.3)
    axes[0,0].legend()
    plt.tight_layout(); save_fig('noise_robustness.png'); plt.close(fig)
