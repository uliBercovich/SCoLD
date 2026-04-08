## SCoLD: Sample Correction of Linkage Disequilibrium

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This repository contains the official Python implementation for the paper **"Calibration improves estimation of linkage disequilibrium on low sample sizes"**.

## Description

Linkage disequilibrium (LD) is commonly measured by the squared correlation ($r^2$) between pairs of genetic variants.
A key issue is that the standard sample estimator is **upward biased** for small sample sizes, especially when true LD is near zero.

This software, **SCoLD**, introduces a simulation-based, model-free calibration procedure to improve LD estimation at small sample sizes (e.g., $n=5,10,25$).
The method uses forward simulation to generate genotype pairs under known allele frequencies and true population $r^2$, then performs an inverse mapping from observed estimates to calibrated values; an optional second step applies a mean-centering correction under independence.

## Features

- **Multiple $r^2$ estimators:** Includes the standard sample estimator (`r2T`) and several sample-size-aware alternatives such as Bulik–Sullivan (https://doi.org/10.1038/ng.3211) and Ragsdale and Gravel (https://doi.org/10.1093/molbev/msz265) estimators (implemented in this codebase/notebook).
- **Non-parametric calibration:** Precomputes calibration curves on a grid of allele-frequency pairs and true $r^2$ values using repeated genotype simulation, then uses inverse interpolation for fast calibration at runtime.
- **Two-step variants:** Supports “general” calibration, “independence” calibration, and “mean-correction” calibration; produces calibrated estimator functions like `r2Tc`, `r2Tic`, `r2Tmc` and analogs for other base estimators.
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

1) choose a base estimator (e.g. `r2T`, `r2BS`, `r2Rag`, …),
2) build calibration models by simulation,
3) create calibrated estimator callables and apply them to genotype pairs.

A minimal example (illustrative; adjust import paths to your package layout):

```python
import numpy as np

# Example: dictionary of estimators (names are used as keys downstream)
r2_estimators = {
    "r2T": r2T,
    "r2BS": r2BS,
    "r2Rag": r2Rag,
    "r2Ber": r2Ber,
}

# 1) Build calibration models for selected sample sizes
n_values = [5, 10, 25]
r2_grid = np.linspace(0, 1, 21)

# This produces a nested dict keyed by estimator -> n -> (MAC-pair) -> (true r2 bin -> mean observed)
master_models = {}
for n in n_values:
    models_for_n = buildcalibrationmodels(
        n=n,
        Nrep=5000,
        r2grid=r2_grid,
        estimatorsdict=r2_estimators,
        njobs=-1,
    )
    for name, model in models_for_n.items():
        master_models.setdefault(name, {})[n] = model

# 2) Create calibrated estimators (general / independence / mean-correction)
r2Tc  = createcalibratedestimator(r2T,  "r2T",  master_models, "general")
r2Tic = createcalibratedestimator(r2T,  "r2T",  master_models, "independence")
r2Tmc = createcalibratedestimator(r2T,  "r2T",  master_models, "meancorrection")

# 3) Apply to a genotype pair matrix G of shape (n_individuals, 2)
# r2_est = r2Tc(G)
```

The calibration functions fall back to the uncalibrated estimator when a model is not available for a given sample size / MAC bin.[^19_2]

## Data

The paper and notebooks evaluate calibration on:

1. **1000 Genomes Project (CEU, chr22):** Accessed via `magenpy` in the analysis notebooks.
2. **Simulated data (AFR demographic model):** Generated with `stdpopsim`/`msprime` for chromosome 22 regions and filtered by MAF.

## Reproducing Figures from the Paper

The `paper/` directory contains Jupyter notebooks to reproduce the main results, including:

- RMSE/bias/variance by distance bins across sample sizes.
- LD pruning evaluation (classification via F1 score) across LD thresholds.

**Note:** This is a working progress.

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
