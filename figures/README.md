# Figures Directory

This folder is git-ignored except for this README and any curated `example_keep/` subfolder.

On running experiments (e.g. `python ds.py --dataset highly_nonlinear --kernel auto`), timestamped result directories like:

```text
highly_nonlinear_20250913_004559/
```

are generated automatically containing:

* performance_comparison.(svg|pdf|png)
* feature_embeddings.(svg|pdf|png)
* correlation_heatmap.(svg|pdf|png)
* performance_summary_overview.(svg|pdf|png)
* performance_summary.(txt|json)

To keep repository size small, these are ignored. If you wish to showcase a minimal example, copy a selected directory into `figures/example_keep/` (which is allowed by `.gitignore`).
