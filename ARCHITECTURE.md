# StreamKSDR Architecture

## 1. Overview

StreamKSDR provides supervised dimensionality reduction with both online (streaming, one‑sample update) and batch variants using Random Fourier Features (RFF). The online algorithm incrementally maintains a projection that maximizes supervised correlation-like structure between transformed inputs and targets.

## 2. Core Modules

| Module | Purpose | Key Notes |
|--------|---------|----------|
| `mainFunction/OKS_main.py` | OnlineKernelSDR implementation | RFF mapping + incremental update of projection matrix U |
| `mainFunction/OKS_batch.py` | BatchKSPCA baseline | Fits projection using entire dataset once |
| `data_gens/` | Synthetic data generators | Each exposes a function used via `get_generator(name)` |
| `kernel_selector.py` or `*_minimal.py` | Simple heuristic kernel choice | Returns 'linear' or 'rbf' based on correlation heuristics |
| `optimization.py` | Lightweight hyperparameter search | Skipped automatically for linear kernel |
| `visualization.py` | Small, fast plots | Embeddings / correlation / performance summary |
| `reporting.py` | Comprehensive & noise analyses | Multi‑panel, robustness & noise structure plots |
| `fig_utils.py` | Figure directory + save helpers | Timestamped run folder + format control |
| `ds.py` | CLI experiment orchestrator | Modes: single/all/noise/benchmark/xor |

## 3. Data / Feature Flow

```text
Raw (X,Y)
  -> (optional) Kernel selection ('linear' passthrough or 'rbf' RFF)
  -> Feature extraction:
        - OnlineKernelSDR: streaming update loop
        - BatchKSPCA: batch fit
  -> Feature matrices (Z_online, Z_batch, PCA, Raw)
  -> Downstream evaluation (classification / regression / correlation)
  -> Ranking + summary
  -> Visualization (visualization.py / reporting.py)
```

## 4. Online Algorithm (High-Level)

1. Map X,Y to random Fourier feature spaces (if kernel != linear; else identity).
2. Maintain running means for centering.
3. Update cross-covariance estimator incrementally.
4. Perform a low-rank update / orthonormalization to keep top-k supervised directions.
5. Project incoming samples using learned U.

## 5. Kernels

- Linear: identity mapping (no sigma, no random projection, faster, no optimization).
- RBF: uses median heuristic / small grid in selection or optimization path.

## 6. Logging

- Unified via Python `logging` (INFO default, WARNING under `--quiet`, DEBUG under `--verbose`).
- Avoid direct `print` in new code; use `logger = logging.getLogger(__name__)`.

## 7. Experiment Modes (`ds.py`)

| Mode | Description |
|------|-------------|
| `dataset` / `single` | One dataset end‑to‑end run |
| `all` | Iterate through all synthetic datasets |
| `noise` | Adds noise structure & robustness analyses |
| `benchmark` | 8 synthetic + kin8nm quick summary (lightweight) |
| `xor` | Convenience alias for `better_xor` |

## 8. Adding a New Dataset

1. Create `data_gens/<name>.py` with a generator function signature `(n_samples, noise_level, random_state, **kwargs)`.
2. Register it in `data_gens/__init__.py` inside the `get_generator` mapping.
3. Run: `python ds.py --dataset <name> --kernel auto`.

## 9. Adding a New Visualization

- Small / per-dataset: implement in `visualization.py` and call from `ds.py` pipeline.
- Complex / multi-figure or noise related: put in `reporting.py`.

## 10. Extending Kernels

- Implement mapping (or identity) in a new utility or extend existing RFF file.
- Update kernel selector if heuristic should consider it (optional).
- Add conditional branches in feature extraction where needed.

## 11. Hyperparameter Optimization

- Only meaningful for non-linear kernels; linear path returns early.
- Keep searches small (max_configs) to avoid heavy runtime.

## 12. Testing (pytest minimal set)

- `tests/test_online_basic.py`: shape validation + no-crash for online/batch models
- `tests/test_kernel_selector.py`: kernel selection sanity checks on linear vs nonlinear data

Run via: `python -m pytest -q`

## 13. Future Improvements

- More rigorous statistical tests of embedding quality and convergence.
- Additional downstream benchmarks beyond classification/regression.
- Performance profiling and optimization for large-scale data.

## 14. Style Guidelines

- Prefer pure functions where practical inside generators.
- Avoid large monolithic plotting blocks inside `ds.py` (route to modules).
- Logging only—no new monkey-patching of `print`.

---

This architecture document is intentionally concise; expand sections if deeper algorithmic exposition is required.
