# Everywhere Valid Bounds on False Discovery Proportions in Conformal Inference

> **https://arxiv.org/abs/2605.20726**

## Overview

This repository contains the code to reproduce all experiments and figures from the paper *"Everywhere Valid Bounds on False Discovery Proportions in Conformal Inference"*.

The paper develops methods for constructing **uniform upper confidence bounds on the False Discovery Proportion (FDP)** that are valid simultaneously over all rejection thresholds, in the setting of conformal inference. 

---

## Requirements

```bash
pip install numpy scipy scikit-learn matplotlib pandas
```

---

## Repository Structure

```
.
├── ecdf_upper_bound.py          # Core module: uniform ECDF and FDP upper bounds
├── ecdf_upper_bound_plot.ipynb  # Visualize ECDF upper bounds (Figs. 2, 8, 9)
├── drug_target_interaction.ipynb# Drug-target interaction experiment (Figs. 1, 6)
├── outlier_simulation.ipynb     # Run outlier detection simulations → experiment_result/
├── outlier_plots.ipynb          # Plot outlier simulation results (Figs. 3, 4, 5)
├── cpvals_ecdf.py               # Utility script for conformal p-value ECDF analysis
│
├── cccpv/                       # Conformal calibration module (CCCPV methods)
│   ├── __init__.py
│   ├── ecdf_upper_bound.py      # Copy of root ecdf_upper_bound.py for package use
│   ├── methods.py               # Calibration methods: Simes, DKWM, Linear, MC, Asymptotic
│   ├── models.py                # ToyModel, ConformalPvalues, CalibrationBound
│   ├── utils_calib.py           # Calibration utilities
│   └── cccpv.ipynb              # CCCPV experiments (Figs. 11, 12)
│
├── supplementary/
│   ├── BH_example.ipynb         # BH procedure illustration (Fig. 7)
│   └── variance_visualization.ipynb  # Variance visualization (Fig. 10)
│
├── experiment_result/           # Saved simulation outputs (.npz)
├── all_figures/                 # Final paper figures (.pdf)
│
├── 199.csv                      # Drug-target interaction dataset
├── calib_199.csv                # Calibration split of the drug-target data
├── test_199.csv                 # Test split of the drug-target data
└── fdp_drug-target-interaction_results.csv  # Saved results for Fig. 6
```

> `TransductiveAdaptive_CP/` is an external dependency (cloned separately); it provides the transductive adaptive conformal prediction baseline.

---

## Reproducing Paper Figures

| Figure | Notebook | Notes |
|--------|----------|-------|
| Fig. 1 (intro) | `drug_target_interaction.ipynb` | |
| Fig. 2 (ECDF bounds) | `ecdf_upper_bound_plot.ipynb` | Default settings |
| Fig. 3–5 (outlier) | `outlier_plots.ipynb` | Run `outlier_simulation.ipynb` first to generate `experiment_result/` |
| Fig. 6 (drug-target) | `drug_target_interaction.ipynb` | |
| Fig. 7 (BH example) | `supplementary/BH_example.ipynb` | |
| Fig. 8–9 (ECDF, varying m/n) | `ecdf_upper_bound_plot.ipynb` | Change `m`, `n`, `setting='iid'` |
| Fig. 10 (variance) | `supplementary/variance_visualization.ipynb` | |
| Fig. 11–12 (CCCPV) | `cccpv/cccpv.ipynb` | |

---

## Core Module: `ecdf_upper_bound.py`

### `uniform_ecdf_upper_bound(n, m, method, ...)`

Returns a function `f : [0,1] → [0,1]` that is a uniform $(1-\delta)$-confidence upper bound on the ECDF of conformal p-values. Supported methods:

| Method | Description |
|--------|-------------|
| `MC-HC` | Monte Carlo Higher Criticism |
| `MC-THC` | Monte Carlo Truncated Higher Criticism |
| `MC-KS` | Monte Carlo Kolmogorov-Smirnov |
| `MC-BJ` | Monte Carlo Berk-Jones |
| `KS` | Analytic DKW-based KS bound (baseline) |
| `Marginal-MC` | Marginal Monte Carlo bound |

**Key parameters:**
- `n` — number of calibration points
- `m` — number of test points
- `N` — number of Monte Carlo replications (default 100)
- `delta` — confidence level (default 0.05)
- `setting` — `'cpvals'` (oracle conformal p-values) or `'iid'` (i.i.d. uniform)

### `uniform_FDP_upper_bound(pvals, n, m, boost, ...)`

Given observed p-values `pvals`, returns a function `f : [0,1] → [0,1]` that upper-bounds the FDP at every threshold simultaneously. The `boost=True` option applies a tighter refinement that exploits the ordering of p-values.

---

## Citation

```bibtex
@article{song2026everywhere,
  title={Everywhere Valid Bounds on False Discovery Proportions in Conformal Inference},
  author={Song, Ziang and Jin, Ying and Cand{\`e}s, Emmanuel J},
  journal={arXiv preprint arXiv:2605.20726},
  year={2026}
}
```
