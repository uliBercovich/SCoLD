## SCoLD: Sample Correction of Linkage Disequilibrium

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository contains the official Python implementation for the paper **"Calibration improves estimation of linkage disequilibrium on low sample sizes"**.

## Description

Linkage disequilibrium (LD) is commonly measured by the squared correlation ($r^2$) between pairs of genetic variants.
A key issue is that the standard sample estimator is **upward biased** for small sample sizes, especially when true LD is near zero.

This software, **SCoLD**, introduces a simulation-based, model-free calibration procedure to improve LD estimation at small sample sizes (e.g., $n=5,10,25$).
The method uses forward simulation to generate genotype pairs under known allele frequencies and true population $r^2$, then performs an inverse mapping from observed estimates to calibrated values; an optional second step applies a mean-centering correction under independence.

## Features

- **Multiple $r^2$ estimators:** Includes the standard sample estimator (`r2`) and several sample-size-aware alternatives: Bulik–Sullivan (`r2_BS`, https://doi.org/10.1038/ng.3211), Ragsdale and Gravel (`r2_Rag`, https://doi.org/10.1093/molbev/msz265), and a supplementary estimator (`r2_Supp`).
- **Non-parametric calibration:** Precomputes calibration curves on a grid of allele-frequency pairs and true $r^2$ values using repeated genotype simulation, then uses inverse interpolation for fast calibration at runtime.
- **Two-step calibration:** Supports inverse-regression calibration (`cal`), independence calibration (`indep`), and their combination (`cal` → `indep`); produces estimator functions like `r2_cal`, `r2_indep`, `r2_cal_indep` and analogs for other base estimators (e.g. `r2_BS_cal`, `r2_BS_cal_indep`).
- **Arbitrary ploidy and pseudohaploid data:** The simulation backing the calibration takes a `ploidy` argument (`1` haploid, `2` diploid, `4` tetraploid, …) and a `pseudohaploid` flag for the ancient-DNA model where one allele is sampled per site.
- **Downstream evaluation:** Includes bootstrap experiments for RMSE/bias/variance by distance bin and LD pruning evaluation using F1 score, mirroring the analyses described in the paper.


## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/uliBercovich/SCoLD.git
cd SCoLD
```

2. **Create a virtual environment (recommended):**

```bash
# Using conda
conda create -n scold_env python=3.11
conda activate scold_env
```

3. **Install dependencies:**
The required packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```


## Basic Usage

The core workflow is:

1) choose a base estimator (`r2`, `r2_BS`, `r2_Rag` or `r2_Supp`),
2) build calibration models by simulation with `build_calibration_models`,
3) create calibrated estimator callables with `create_calibrated_estimator` and apply them to genotype pairs.

A minimal example:

```python
import numpy as np

from ld_estimates import (
    build_calibration_models,
    create_calibrated_estimator,
    r2, r2_BS, r2_Rag, r2_Supp,
    r2_batch, r2_BS_batch, r2_Supp_batch,
)

# Estimators to calibrate; the keys are used as names downstream
estimators_to_calibrate = {
    "r2":      r2,
    "r2_BS":   r2_BS,
    "r2_Rag":  r2_Rag,
    "r2_Supp": r2_Supp,
}

# Optional vectorised versions; used automatically when provided, and much
# faster. Estimators without a batch version fall back to a scalar loop.
batch_estimators = {
    "r2":      r2_batch,
    "r2_BS":   r2_BS_batch,
    "r2_Supp": r2_Supp_batch,
}

# 1) Build calibration models for the sample sizes of interest.
#    Returns a nested dict: estimator -> n -> MAC pair -> (true r2 -> mean observed r2)
n_values = [5, 10, 25]
r2_grid = np.linspace(0, 1, 21)

master_models = {}
for n in n_values:
    models_for_n = build_calibration_models(
        n=n,
        N_replicates=5000,
        r2_grid_to_model=r2_grid,
        estimators_to_calibrate=estimators_to_calibrate,
        batch_estimators=batch_estimators,
        ploidy=2,           # 1 = haploid, 2 = diploid (default), 4 = tetraploid, ...
        pseudohaploid=False,  # True for the ancient-DNA pseudohaploid model
        n_jobs=-1,
    )
    for name, model in models_for_n.items():
        master_models.setdefault(name, {})[n] = model

# 2) Create the calibrated estimators
r2_cal       = create_calibrated_estimator(r2, "r2", master_models, "cal")
r2_indep     = create_calibrated_estimator(r2, "r2", master_models, "indep")

# The two-step estimator calibrates r2_cal again with the independence step, so
# it needs a calibration model built for "r2_cal" itself (same call as above,
# passing {"r2_cal": r2_cal} as estimators_to_calibrate).
r2_cal_indep = create_calibrated_estimator(r2_cal, "r2_cal", master_models, "indep")

# 3) Apply to a genotype pair matrix G of shape (n_individuals, 2), entries in {0, 1, 2}
# r2_est = r2_cal(G)
```

The calibration functions fall back to the uncalibrated estimator when a model is not available for a given sample size or MAC bin.

### Ploidy and pseudohaploid data

`build_calibration_models`, `apply_calibration` and `create_calibrated_estimator` all take
`ploidy` and `pseudohaploid` arguments, which must match between model building and
application. Use `ploidy=1` for haploid data and `ploidy=4` for tetraploids.

Setting `pseudohaploid=True` simulates the ancient-DNA model in which individuals are
diploid but a single read (allele) is sampled per site, independently across sites. Note
that the naive $r^2$ between pseudohaploid markers is downward biased by a factor of
approximately 4 at any sample size, since sampling one allele per site halves the
covariance $D$; calibrating $4r^2$ rather than $r^2$ therefore puts the estimator back on
the right scale before the small-sample correction is applied.

## Data

The analyses use three datasets, all restricted to chromosome 22 with MAF > 5%:

| Dataset | Source | Ships as | Provenance script |
|---|---|---|---|
| **CEU** | 1000 Genomes EUR | fetched at runtime via `magenpy` | — |
| **AFR** | `stdpopsim` `Africa_1T12` simulation | `paper/data/AFR_chr22.npz` | `paper/gen_afr.py` |
| **Structured** (50 CEU + 50 YRI + 50 ASW) | 1000 Genomes, used for the covariance-matrix calibration | `paper/data/1000G_struct_chr22.npz` | `paper/gen_1000g_struct.py` |

The two `.npz` inputs are committed, so everything except the CEU figures reproduces without
`magenpy`/`stdpopsim`. To regenerate the structured dataset, download the raw 1000 Genomes
phase-3 chr22 VCF and panel and point `G1000_DIR` at them (see the header of `gen_1000g_struct.py`).

## Reproducing the figures

Reproduction has two layers: **compute scripts** turn genotypes into result tables under
`paper/output/`, and **notebooks** turn those tables into the manuscript figures under
`paper/figures/`. The result tables are committed, so you can regenerate any figure by just
running its notebook; rerun a compute script only to reproduce the tables from scratch.

**1. Compute (writes `paper/output/*.csv`, seeded):**

```bash
python paper/run_main_experiments.py --pop AFR      # metrics/f1/ldscore/summary (add --quick to smoke-test)
python paper/run_main_experiments.py --pop CEU      # needs magenpy
python paper/run_pseudohaploid_by_distance.py       # pseudo_metrics.csv          (S4)
python paper/run_adjusted_calibration.py            # adjcal_metrics.csv           (S5)
```

**2. Figures — run the notebooks in `paper/notebooks/`:**

| Notebook | Figures |
|---|---|
| `main_figures.ipynb` | Fig. 2 (bias/RMSE/variance), Fig. 4 (F1), Supp. S3 (LD score), Tables S1–S2 |
| `pseudohaploid.ipynb` | Supp. S4 (`pseudo_bias.pdf`, `pseudo_rmse.pdf`) |
| `covariance_calibration.ipynb` | Supp. S5 (`adjcal_both.pdf`) |
| `flex.ipynb` | Flex analysis (not in the paper; kept for completeness) |

`experiments.py` and `plotting.py` hold the shared compute and plotting helpers the scripts and
notebooks call.

**Note on preserved outputs.** The original F1-plotting and LD-pruning-histogram code was not
kept, so the F1 figure (Fig. 4) is reconstructed from the sampled pairs in
`compute_f1_by_threshold`, and the pruning histogram (Fig. 3) and the calibrated-competitor
comparison (Supp. `*_calibrating_methods`) are not regenerated by this repository.

## Citation

If you use this code or method in your research, please cite our paper:

```bibtex
@article{Bercovich2026SCoLD,
  author  = {Bercovich, Ulises and Wiuf, Carsten and Albrechtsen, Anders},
  title   = {Calibration improves estimation of linkage disequilibrium on low sample sizes}
  % TODO: Add Journal/Year/DOI when available
}
```


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
