"""Ancestry-adjusted LD (Bercovich et al. 2025) and its sample-size calibration.

The adjusted estimator removes the LD induced by population structure by correlating
genotype residuals after projecting out the top k principal components:

    2*Pi_hat = P_hat G,   P_hat = projection onto the mean + top k PCs
    R        = G - 2*Pi_hat
    D_adj    = R^T (I - J/n) R / (2(n - k))
    r2_adj   = D_adj[s,t]^2 / (D_adj[s,s] D_adj[t,t])

IMPORTANT: P_hat is fitted on the *whole* genotype matrix (all m variants) and then
applied to each pair. It is not a function of a single SNP pair -- fitting it on two
columns would span their entire column space and leave zero residual.

This removes the *structure* bias, which does not vanish as n grows. It does not remove
the *sample-size* bias, which is still present because r2_adj is a ratio of estimated
covariances.

Calibrating r2_adj:
  the projection preserves column means, so the marginal allele frequency survives the
  residual step untouched -- but it is no longer the right quantity to condition on. The
  residual variance is the *within*-population heterozygosity,

      D_adj[s,s] ~= p_s(1-p_s) (1 - F_ST,s)  <  p_s(1-p_s),

  so we read an effective frequency off the adjusted variance itself,

      p_eff(1 - p_eff) = D_adj[s,s],

  and calibrate with the existing homogeneous curves at that frequency and at an
  effective sample size n_eff = n - k (the residual degrees of freedom).
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def fit_residuals(G: np.ndarray, k: int) -> np.ndarray:
    """Genome-wide residual matrix R = G - 2*Pi_hat, for an n x m genotype matrix.

    Uses mean-centred PCA: the fitted value is the mean plus the projection onto the top
    k principal components, so the residuals have mean zero per column and the marginal
    allele frequency is unchanged by the residual step.
    """
    mu = G.mean(axis=0, keepdims=True)
    Gc = G - mu
    if k <= 0:
        return Gc
    U, _, _ = np.linalg.svd(Gc, full_matrices=False)
    k = min(k, U.shape[1])
    Uk = U[:, :k]
    return Gc - Uk @ (Uk.T @ Gc)


def adjusted_D_from_residuals(R_pair: np.ndarray, n: int, k: int) -> np.ndarray:
    """2x2 adjusted covariance for two residual columns (n x 2), with the (n-k) correction."""
    Rc = R_pair - R_pair.mean(axis=0, keepdims=True)
    return (Rc.T @ Rc) / (2.0 * (n - k))


def r2_adj_from_residuals(R_pair: np.ndarray, n: int, k: int) -> float:
    """Adjusted r^2 from two residual columns. NaN if either site has no residual variance."""
    D = adjusted_D_from_residuals(R_pair, n, k)
    denom = D[0, 0] * D[1, 1]
    if not np.isfinite(denom) or denom <= 0:
        return np.nan
    return float(D[0, 1] ** 2 / denom)


def effective_freq(var_adj: float) -> float:
    """Solve p(1-p) = var_adj for p in [0, 0.5]: the effective (within-population) frequency.

    var_adj is the adjusted per-site variance D_adj[s,s].
    """
    if not np.isfinite(var_adj) or var_adj <= 0:
        return np.nan
    v = min(float(var_adj), 0.25)        # p(1-p) <= 1/4
    return float(0.5 * (1.0 - np.sqrt(max(0.0, 1.0 - 4.0 * v))))


def effective_mac(var_adj: float, n_eff: int, ploidy: int = 2) -> Optional[int]:
    """Effective minor-allele count for the calibration lookup: round p_eff onto the MAC grid."""
    p = effective_freq(var_adj)
    if not np.isfinite(p):
        return None
    mac = int(round(p * ploidy * n_eff))
    return int(np.clip(mac, 1, max(1, ploidy * n_eff // 2)))
