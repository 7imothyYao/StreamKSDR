# StreamKSDR (Online Kernel Supervised Dimensionality Reduction)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
![CI](https://github.com/your-org/StreamKSDR/actions/workflows/ci.yml/badge.svg)

This repository provides online (streaming, one-sample updates) and batch variants
of Kernel Supervised Dimensionality Reduction (KSDR) using Random Fourier Features (RFF).
Linear kernel is now a strict identity (no random projection, no sigma search). A minimal
kernel selector optionally chooses between 'linear' and 'rbf' via a lightweight correlation heuristic.

* `OnlineKernelSDR` (formerly `RealtimeOnlineKSPCA`): single-sample streaming updates with adaptive learning rate / optional advanced adaptation, forgetting factor, optional whitening, stability guards.
* `BatchKSPCA`: batch reference implementation (RFF approximation).
* `data_gens/`: nonlinear synthetic generators for benchmarking and visualization.
* `kin8nm_experiment.py`: nested CV experiment on the UCI kin8nm dataset (path fixed to `data_real/kin8nm/`).
* `ds.py`: combined experiment / visualization driver (multi-dataset automation + downstream baselines). Hyperparameter search is automatically skipped for linear kernel.
* `kernel_selector_minimal.py`: tiny heuristic selector (pure NumPy) — can be ignored if you choose kernels manually.

## Quick Start

* Install dependencies (Python >= 3.10):

```bash
pip install -r requirements.txt
```

* Run an online synthetic dataset experiment (auto kernel selection):

```bash
python ds.py --dataset highly_nonlinear --kernel auto
```

* Run the real kin8nm experiment (rbf kernel example):

```bash
python kin8nm_experiment.py --data-path data_real/kin8nm/dataset_2175_kin8nm.arff
```

## Core Classes

* Online: `mainFunction/OKS_main.py`
* Batch: `mainFunction/OKS_batch.py`

## Minimal Smoke Test (Manual)

One‑liner to verify core online + batch paths (linear identity, fast):

```bash
python ds.py --dataset better_xor --kernel linear --no-big-plots --fig-formats png
```

RBF quick check:

```bash
python ds.py --dataset better_xor --kernel rbf --no-big-plots --fig-formats png
```

Optional: subspace comparison (produces a PNG):

```bash
python subspace_comparison_experiment.py
```

## Typical Outputs

Experiments create timestamped folders under `figures/` storing:

* Learned embedding scatter plots / correlation heatmaps
* Downstream evaluation tables (classification / regression)
* Optional online training metrics

## Multi-Dataset & Benchmark Modes

Unified driver: `ds.py` (no separate run_all_datasets.py). Use `--mode`:

| Mode | Example Command | Description |
|------|-----------------|-------------|
| (default) | `python ds.py --dataset highly_nonlinear --kernel auto` | Single dataset run |
| all | `python ds.py --mode all --kernel auto --no-big-plots` | Loop over all synthetic generators |
| noise | `python ds.py --mode noise --dataset highly_nonlinear --kernel auto` | Adds noise structure & robustness analyses |
| benchmark | `python ds.py --mode benchmark --kernel auto --no-big-plots --no_optimize` | 8 synthetic + kin8nm quick summary |

Flags:

* `--quiet` suppress routine logs (keeps warnings/errors/final summaries)
* `--verbose` extra diagnostics (ignored if quiet)
* `--no-big-plots` skip heavy multi‑panel composites
* `--fig-formats png,svg,pdf` select formats (default png)
* `--no_optimize` skip hyperparameter search (linear always skips)

Per-dataset folder under `figures/` typically contains:

* performance_comparison.*
* feature_embeddings.*
* correlation_heatmap.*
* performance_summary.(txt|json)

Noise mode adds robustness / noise structure plots.

Minimal single-dataset (auto selection):

```bash
python ds.py --dataset highly_nonlinear --kernel auto
```

Linear baseline:

```bash
python ds.py --dataset highly_nonlinear --kernel linear
```

## Dataset Note

`data_real/kin8nm/dataset_2175_kin8nm.arff` must be obtained from the UCI repository
and placed accordingly. Unused `sarcos_data` assets were removed.

## Notes / Design Decisions

* Linear kernel: pure passthrough (no random features). Dimension must match input. Hyperparameter search is skipped.
* RFF implementation unified in `mainFunction/rff.py`.
* Online and batch share the same feature mapping; projection learning differs.
* Kernel selector is optional; you can supply `--kernel linear` or `--kernel rbf` directly.
* Figures: specify formats (e.g. `--fig-formats png`). Large multi‑panel plots disabled with `--no-big-plots`.
* Optimization: only meaningful for non-linear kernels; automatically bypassed for linear to avoid noise.

## Testing

Basic regression tests available:

```bash
pip install pytest
python -m pytest -q
```

Tests cover core algorithm sanity, kernel selector behavior, and basic functionality.

## Suggested Next Enhancements

| Area | Idea |
|------|------|
| Packaging | Add `pyproject.toml` for modern Python packaging |
| Benchmark Config | YAML/JSON profile for consistent multi-run reproducibility |
| Caching | Memoize adaptive sigma search results per dataset/kernel |
| Advanced Tests | Statistical validation of embedding quality |

## License

Released under the MIT License (see `LICENSE`).

---
Issues / PRs welcome for improvements: online stability, adaptive strategies, new kernels.
