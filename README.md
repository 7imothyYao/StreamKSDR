# StreamKSDR (Online Kernel Supervised Dimensionality Reduction)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Online + streaming (single-sample) and batch variants of Kernel Supervised Dimensionality Reduction (KSDR) using Random Fourier Features (RFF). Linear kernel is a strict identity (no random projection, no sigma search). A lightweight heuristic kernel selector (`kernel_selector.py`) computes a combined linearity score (max feature‑target correlation + simple explained variance mix) to choose `linear` vs `rbf` when `--kernel auto`.

* `OnlineKernelSDR` (formerly `RealtimeOnlineKSPCA`): streaming updates (optional adaptive lr / forgetting / stability guards)
* `BatchKSPCA`: batch reference (same RFF mapping interface)
* `data_gens/`: synthetic generators for benchmarking & visualization
* `ds.py`: unified experiment / visualization / multi-dataset automation driver
* `kin8nm_experiment.py`: real regression dataset example (UCI kin8nm)
* `kernel_selector.py`: heuristic kernel selection (combined linearity score)

See `ARCHITECTURE.md` for internal design, data flow, and extension points.

---

## 1. Quick Start

Python >= 3.10

```bash
pip install -r requirements.txt
```

Minimal run (auto kernel, one synthetic dataset):

```bash
python ds.py --dataset highly_nonlinear --kernel auto
```

Real dataset (download ARFF first – see Dataset section):

```bash
python kin8nm_experiment.py --data-path data_real/kin8nm/dataset_2175_kin8nm.arff
```

Fast sanity (linear identity path):

```bash
python ds.py --dataset better_xor --kernel linear --no-big-plots --fig-formats png
```

---

## 2. Datasets & Modes

Synthetic datasets (8):

| Name | Input Dim (approx) | Targets | Complexity | Notes |
|------|--------------------|---------|------------|-------|
| highly_nonlinear | 10 | 5 | Medium | Spiral + radial + polynomial mix |
| better_xor | 101 | 5 | High | Multi-output smooth XOR + interactions |
| nuclear_friendly | 208 | 5 | High | Low linear corr, strong nonlinear deps |
| extreme1 | 60 | 5 | High | Rich composite nonlinear transforms |
| extreme2 | 80 | 5 | Very High | Deeper composite structure |
| extreme3 | 100 | 5 | Very High | Torus / harmonic style structure |
| piecewise | 50 | 5 | Medium | Discontinuities + conditional branches |
| swiss | 80 | 5 | High | Swiss roll manifold embedding |

Modes (`ds.py --mode ...`):

| Mode | Description | Example |
|------|-------------|---------|
| (default) | Single dataset run | `python ds.py --dataset highly_nonlinear --kernel auto` |
| all | Loop over all synthetic datasets | `python ds.py --mode all --kernel auto --no-big-plots` |
| noise | Adds noise structure / robustness plots | `python ds.py --mode noise --dataset highly_nonlinear --kernel auto` |
| benchmark | 8 synthetic + kin8nm quick summary | `python ds.py --mode benchmark --kernel auto --no-big-plots --no_optimize` |
| xor | Shortcut for better_xor only | `python ds.py --mode xor --kernel auto` |

Common flags:

* `--quiet` (suppress routine logs; keeps warnings & final summary)
* `--verbose` (extra diagnostics; ignored if quiet)
* `--no-big-plots` (skip heavy multi‑panel figures)
* `--fig-formats png,svg,pdf` (comma list; default png)
* `--no_optimize` (disable hyperparameter search)

---

## 3. Running Examples

Single RBF check:

```bash
python ds.py --dataset better_xor --kernel rbf --no-big-plots
```

Auto kernel on each synthetic dataset:

```bash
python ds.py --dataset highly_nonlinear --kernel auto
python ds.py --dataset better_xor --kernel auto
python ds.py --dataset nuclear_friendly --kernel auto
python ds.py --dataset extreme1 --kernel auto
python ds.py --dataset extreme2 --kernel auto
python ds.py --dataset extreme3 --kernel auto
python ds.py --dataset piecewise --kernel auto
python ds.py --dataset swiss --kernel auto
```

All synthetic (batch run):

```bash
python ds.py --mode all --kernel auto --no-big-plots
```

Benchmark (synthetic + kin8nm subset):

```bash
python ds.py --mode benchmark --kernel auto --no-big-plots --no_optimize
```

Noise robustness example:

```bash
python ds.py --mode noise --dataset highly_nonlinear --kernel auto
```

Subspace comparison visualization:

```bash
python subspace_comparison_experiment.py
```

---

## 4. Outputs & Metrics

Each run creates a timestamped folder under `figures/` containing (extensions per selected formats):

* `feature_embeddings.*` – low‑dim embedding scatter
* `correlation_heatmap.*` – feature–target correlation proxy
* `performance_comparison.*` – comparison of raw / PCA / batch / online
* `performance_summary.txt|json` – machine-readable metrics

`performance_summary` includes (typical keys):

* `classification_acc` – downstream balanced/standard accuracy (scaled 0–1)
* `regression_r2` – predictive R2 (may be negative for poor fit)
* `avg_correlation` – mean abs correlation between extracted components & targets
* `feature_dim` – effective latent dimension (k) used in scoring penalty

---

## 5. Hyperparameters & Optimization

Current built‑in search: **preset candidate lists (per dataset)** + optional random down‑sampling when candidates > `max_configs`. It is *not* a full grid or Bayesian search. Linear kernel path is skipped entirely (identity mapping). Use `--no_optimize` to disable.

Process (non-linear kernels):

1. Optional auto kernel selection (`--kernel auto`) → `linear` vs `rbf` (combined linearity score).
2. Load a small curated list of candidate configs (hand-tuned ranges for `(D_x, D_y, sigma_x, sigma_y, learning_rate)`).
3. If list longer than `max_configs`, randomly sample a subset without replacement.
4. Evaluate each once; compute composite score:
   `0.4*classification_acc + 0.4*max(0, regression_r2) + 0.2*avg_correlation`, scaled by `1/(1 + feature_dim/10)`.
5. Best configuration is reported (currently NOT auto-applied to the subsequent full run – integration placeholder).

Disable all search:

```bash
python ds.py --dataset highly_nonlinear --kernel rbf --no_optimize
```

---

## 6. Pytest Regression Tests

Install & run:

```bash
pip install pytest
python -m pytest -q
```

Covers: core streaming update sanity, kernel selector behavior, projection consistency. Runtime is small (fast smoke coverage, not exhaustive statistical validation).

---

## 7. Internals

Core classes:

* Online: `mainFunction/OKS_main.py`
* Batch: `mainFunction/OKS_batch.py`

Random feature mapping unified in `mainFunction/rff.py`. Design & extension notes (adding kernels, new generators, adaptation strategies) are detailed in `ARCHITECTURE.md`.

---

## 8. Dataset Notes

Real kin8nm file: place `dataset_2175_kin8nm.arff` under `data_real/kin8nm/` (download from UCI Machine Learning Repository). Other legacy `sarcos_data` assets removed.

---

## 9. Suggested Next Enhancements

| Area | Idea |
|------|------|
| Packaging | Add `pyproject.toml` (modern packaging / build isolation) |
| Benchmark Config | YAML/JSON profile for reproducible multi-run suites |
| Caching | Memoize sigma search results per dataset/kernel |
| Advanced Tests | Statistical embedding quality assertions |
| New Kernels | Add Laplace / Matern tunable variants in optimization |

---

## 10. License

Released under the MIT License (see `LICENSE`).

---
Issues / PRs welcome (focus ideas: online stability, adaptive strategies, new kernels).
