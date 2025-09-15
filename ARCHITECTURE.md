# StreamKSDR Architecture

## 1. Overview

StreamKSDR provides supervised dimensionality reduction with both online (streaming, one‑sample update) and batch variants using Random Fourier Features (RFF). The online algorithm incrementally maintains a projection that maximizes supervised correlation-like structure between transformed inputs and targets.

## 2. Core Modules

| Module | Purpose | Key Notes |
|--------|---------|----------|
| `mainFunction/OKS_main.py` | OnlineKernelSDR implementation | RFF mapping + incremental update of projection matrix U |
| `mainFunction/OKS_batch.py` | BatchKSPCA baseline | Fits projection using entire dataset once |
| `data_gens/` | Synthetic data generators | Each exposes a function via `get_generator(name)` |
| `kernel_selector.py` | Heuristic kernel selection | Combined linearity score (max corr + simple explained var) → linear vs rbf |
| `optimization.py` | Optional preset search | Curated candidate list; skipped for linear; not auto-applied yet |
| `visualization.py` | Small, fast plots | Embeddings / correlation / performance comparison (compact) |
| `reporting.py` | Extended & noise analyses | Multi‑panel overview, learning curves, noise robustness |
| `fig_utils.py` | Figure directory + save helpers | Timestamped run folder + format control |
| `model_utils.py` | Projection helper | `project_features` for consistent centering + projection |
| `ds.py` | CLI experiment orchestrator | Modes: dataset/all/noise/benchmark/xor |

## 3. Data / Feature Flow

```text
Raw (X,Y)
  -> (optional) kernel selection (auto: linear passthrough or rbf RFF)
  -> Feature extraction:
        OnlineKernelSDR (streaming) / BatchKSPCA (batch)
  -> Feature matrices (Z_online, Z_batch, PCA, Raw)
  -> Downstream evaluation (classification / regression / correlation)
  -> Scoring + summary (dimension penalty)
  -> Visualization (visualization.py + reporting.py)
```

## 4. Online Algorithm (High-Level)

1. Map X,Y to random Fourier feature spaces (if kernel != linear; else identity)
2. Maintain running means for centering
3. Incrementally update cross-covariance estimator
4. Low-rank orthonormalization / top-k update
5. Project new samples via learned U

## 5. Kernels

- Linear: identity mapping (no sigma, no random projection, no search)
- RBF: random Fourier features; sigma chosen from small preset candidates (or defaults)

## 6. Logging

- Standard Python `logging` (INFO default, WARNING under `--quiet`, DEBUG under `--verbose`)
- No monkey-patching of `print`

## 7. Experiment Modes (`ds.py`)

| Mode | Description |
|------|-------------|
| `dataset` / `single` | One dataset run |
| `all` | Iterate all synthetic datasets |
| `noise` | Adds noise structure + robustness analyses |
| `benchmark` | 8 synthetic + kin8nm quick summary |
| `xor` | Convenience alias for `better_xor` |

## 8. Adding a New Dataset

1. Create `data_gens/<name>.py` with `(n_samples, noise_level, random_state, **kwargs)`
2. Register in `data_gens/__init__.py`
3. Run: `python ds.py --dataset <name> --kernel auto`

## 9. Visualization Layers

- `visualization.py`: compact single-purpose figures (embeddings, correlation heatmap, performance bar charts)
- `reporting.py`: comprehensive multi-panel overview, noise structure, noise robustness, learning curves

## 10. Kernel Selection Heuristic

- Score = 0.7 \* (max feature–target correlation aggregate) + 0.3 \* (simple explained variance on first target)
- Threshold (default 0.4): score >= threshold → linear else rbf
- Confidence is a clipped scaled function of score distance

## 11. Hyperparameter Optimization (Optional)

- Not a grid or adaptive search; uses a **curated candidate list per dataset**
- If candidates > `max_configs`, random subset sampled without replacement
- Composite score: `0.4*classification_acc + 0.4*max(0, regression_r2) + 0.2*avg_correlation`, scaled by `1/(1 + feature_dim/10)`
- Linear path short-circuits to identity config
- Best config currently **reported but not auto-applied** (hook placeholder)

## 12. Testing (pytest minimal set)

- `tests/test_online_basic.py`: shape + streaming/batch sanity
- `tests/test_kernel_selector.py`: kernel heuristic behavior

Run: `python -m pytest -q`

## 13. Future Improvements

- Statistical validation of embedding quality
- Additional downstream tasks / benchmarks
- Performance profiling (large-scale streaming)
- Automatic application of best hyperparameters

## 14. Style Guidelines

- Prefer modular small plotting functions
- Keep heavy analysis in `reporting.py`
- Use logging only; avoid silent broad exceptions

---

Concise by design; extend with deeper derivations if needed.
