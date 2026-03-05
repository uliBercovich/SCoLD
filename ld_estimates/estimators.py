# estimators.py
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict
import numpy as np


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def r2_T(G: np.ndarray) -> float:
    """
    Naive sample r^2 = corr(G[:,0], G[:,1])^2 using numpy.
    Expects G shape (n,2) with entries in {0,1,2}.
    """
    r = np.corrcoef(G.T)[0, 1]
    return float(r * r)


def r2_BS(G: np.ndarray, cut: bool = False) -> float:
    """
    Bulik-Sullivan correction: r2 - (1-r2)/(n-2) (as in your notebook).
    """
    n = int(G.shape[0])
    r2 = r2_T(G)
    out = r2 - (1.0 - r2) / (n - 2)
    return _clip01(out) if cut else float(out)


# ---- Ragsdale-Gravel pieces (unphased 0/1/2) ----

def _counts_3x3(G: np.ndarray) -> Counter:
    """
    Returns Counter over pairs (g_s, g_t) with g in {0,1,2}.
    """
    return Counter(map(tuple, G.astype(int)))


def D2_Rag(G: np.ndarray, counts: Counter | None = None) -> float:
    """
    Ragsdale and Gravel D^2 correction.
    """
    if counts is None:
        counts = _counts_3x3(G)

    c22 = counts[(2, 2)]
    c21 = counts[(2, 1)]
    c20 = counts[(2, 0)]
    c12 = counts[(1, 2)]
    c11 = counts[(1, 1)]
    c10 = counts[(1, 0)]
    c02 = counts[(0, 2)]
    c01 = counts[(0, 1)]
    c00 = counts[(0, 0)]
    n1, n2, n3, n4, n5, n6, n7, n8, n9 = c22, c21, c20, c12, c11, c10, c02, c01, c00
    n = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9

    numer = (n2*n4 - n2**2*n4 + 4*n3*n4 - 4*n2*n3*n4 - 4*n3**2*n4 - n2*n4**2 - 4*n3*n4**2 + n1*n5 - n1**2*n5 + n3*n5 + 2*n1*n3*n5 - n3**2*n5 - 4*n3*n4*n5 - n1*n5**2 - n3*n5**2 + 4*n1*n6 - 4*n1**2*n6 + n2*n6 - 4*n1*n2*n6 - n2**2*n6 + 2*n2*n4*n6 - 4*n1*n5*n6 - 4*n1*n6**2 - n2*n6**2 + 4*n2*n7 - 4*n2**2*n7 + 16*n3*n7 - 16*n2*n3*n7 - 16*n3**2*n7 - 4*n2*n4*n7 - 16*n3*n4*n7 + n5*n7 + 2*n1*n5*n7 - 4*n2*n5*n7 - 18*n3*n5*n7 - n5**2*n7 + 4*n6*n7 + 8*n1*n6*n7 - 16*n3*n6*n7 - 4*n5*n6*n7 - 4*n6**2*n7 - 4*n2*n7**2 - 16*n3*n7**2 - n5*n7**2 - 4*n6*n7**2 + 4*n1*n8 - 4*n1**2*n8 + 4*n3*n8 + 8*n1*n3*n8 - 4*n3**2*n8 + n4*n8 - 4*n1*n4*n8 + 2*n2*n4*n8 - n4**2*n8 - 4*n1*n5*n8 - 4*n3*n5*n8 + n6*n8 + 2*n2*n6*n8 - 4*n3*n6*n8 + 2*n4*n6*n8 - n6**2*n8 - 16*n3*n7*n8 - 4*n6*n7*n8 - 4*n1*n8**2 - 4*n3*n8**2 - n4*n8**2 - n6*n8**2 + 16*n1*n9 - 16*n1**2*n9 + 4*n2*n9 - 16*n1*n2*n9 - 4*n2**2*n9 + 4*n4*n9 - 16*n1*n4*n9 + 8*n3*n4*n9 - 4*n4**2*n9 + n5*n9 - 18*n1*n5*n9 - 4*n2*n5*n9 + 2*n3*n5*n9 - 4*n4*n5*n9 - n5**2*n9 - 16*n1*n6*n9 - 4*n2*n6*n9 + 8*n2*n7*n9 + 2*n5*n7*n9 - 16*n1*n8*n9 - 4*n4*n8*n9 - 16*n1*n9**2 - 4*n2*n9**2 - 4*n4*n9**2 - n5*n9**2)/16. + (-((n2/2. + n3 + n5/4. + n6/2.)*(n4/2. + n5/4. + n7 + n8/2.)) + (n1 + n2/2. + n4/2. + n5/4.)*(n5/4. + n6/2. + n8/2. + n9))**2
    denom = n * (n - 1) * (n - 2) * (n - 3)
    return float(4.0 * numer / denom)


def Ddiag_Rag(G: np.ndarray, counts: Counter | None = None) -> float:
    """
    Ragsdale and Gravel Var*Var correction.
    """
    if counts is None:
        counts = _counts_3x3(G)

    c22 = counts[(2, 2)]
    c21 = counts[(2, 1)]
    c20 = counts[(2, 0)]
    c12 = counts[(1, 2)]
    c11 = counts[(1, 1)]
    c10 = counts[(1, 0)]
    c02 = counts[(0, 2)]
    c01 = counts[(0, 1)]
    c00 = counts[(0, 0)]
    n1, n2, n3, n4, n5, n6, n7, n8, n9 = c22, c21, c20, c12, c11, c10, c02, c01, c00
    n = n1 + n2 + n3 + n4 + n5 + n6 + n7 + n8 + n9

    numer = (n1 + n2 + n3 + n4/2. + n5/2. + n6/2.)*(n1 + n2/2. + n4 + n5/2. + n7 + n8/2.)*(n2/2. + n3 + n5/2. + n6 + n8/2. + n9)*(n4/2. + n5/2. + n6/2. + n7 + n8 + n9) + (13*n2*n4 - 16*n1*n2*n4 - 11*n2**2*n4 + 16*n3*n4 - 28*n1*n3*n4 - 24*n2*n3*n4 - 8*n3**2*n4 - 11*n2*n4**2 - 20*n3*n4**2 - 6*n5 + 12*n1*n5 - 4*n1**2*n5 + 17*n2*n5 - 20*n1*n2*n5 - 11*n2**2*n5 + 12*n3*n5 - 28*n1*n3*n5 - 20*n2*n3*n5 - 4*n3**2*n5 + 17*n4*n5 - 20*n1*n4*n5 - 32*n2*n4*n5 - 40*n3*n4*n5 - 11*n4**2*n5 + 11*n5**2 - 16*n1*n5**2 - 17*n2*n5**2 - 16*n3*n5**2 - 17*n4*n5**2 - 6*n5**3 + 16*n1*n6 - 8*n1**2*n6 + 13*n2*n6 - 24*n1*n2*n6 - 11*n2**2*n6 - 28*n1*n3*n6 - 16*n2*n3*n6 + 24*n4*n6 - 36*n1*n4*n6 - 38*n2*n4*n6 - 36*n3*n4*n6 - 20*n4**2*n6 + 17*n5*n6 - 40*n1*n5*n6 - 32*n2*n5*n6 - 20*n3*n5*n6 - 42*n4*n5*n6 - 17*n5**2*n6 - 20*n1*n6**2 - 11*n2*n6**2 - 20*n4*n6**2 - 11*n5*n6**2 + 16*n2*n7 - 28*n1*n2*n7 - 20*n2**2*n7 + 16*n3*n7 - 48*n1*n3*n7 - 44*n2*n3*n7 - 16*n3**2*n7 - 24*n2*n4*n7 - 44*n3*n4*n7 + 12*n5*n7 - 28*n1*n5*n7 - 40*n2*n5*n7 - 48*n3*n5*n7 - 20*n4*n5*n7 - 16*n5**2*n7 + 16*n6*n7 - 48*n1*n6*n7 - 48*n2*n6*n7 - 44*n3*n6*n7 - 36*n4*n6*n7 - 40*n5*n6*n7 - 20*n6**2*n7 - 8*n2*n7**2 - 16*n3*n7**2 - 4*n5*n7**2 - 8*n6*n7**2 + 16*n1*n8 - 8*n1**2*n8 + 24*n2*n8 - 36*n1*n2*n8 - 20*n2**2*n8 + 16*n3*n8 - 48*n1*n3*n8 - 36*n2*n3*n8 - 8*n3**2*n8 + 13*n4*n8 - 24*n1*n4*n8 - 38*n2*n4*n8 - 48*n3*n4*n8 - 11*n4**2*n8 + 17*n5*n8 - 40*n1*n5*n8 - 42*n2*n5*n8 - 40*n3*n5*n8 - 32*n4*n5*n8 - 17*n5**2*n8 + 13*n6*n8 - 48*n1*n6*n8 - 38*n2*n6*n8 - 24*n3*n6*n8 - 38*n4*n6*n8 - 32*n5*n6*n8 - 11*n6**2*n8 - 28*n1*n7*n8 - 36*n2*n7*n8 - 44*n3*n7*n8 - 16*n4*n7*n8 - 20*n5*n7*n8 - 24*n6*n7*n8 - 20*n1*n8**2 - 20*n2*n8**2 - 20*n3*n8**2 - 11*n4*n8**2 - 11*n5*n8**2 - 11*n6*n8**2 + 16*n1*n9 - 16*n1**2*n9 + 16*n2*n9 - 44*n1*n2*n9 - 20*n2**2*n9 - 48*n1*n3*n9 - 28*n2*n3*n9 + 16*n4*n9 - 44*n1*n4*n9 - 48*n2*n4*n9 - 48*n3*n4*n9 - 20*n4**2*n9 + 12*n5*n9 - 48*n1*n5*n9 - 40*n2*n5*n9 - 28*n3*n5*n9 - 40*n4*n5*n9 - 16*n5**2*n9 - 44*n1*n6*n9 - 24*n2*n6*n9 - 36*n4*n6*n9 - 20*n5*n6*n9 - 48*n1*n7*n9 - 48*n2*n7*n9 - 48*n3*n7*n9 - 28*n4*n7*n9 - 28*n5*n7*n9 - 28*n6*n7*n9 - 44*n1*n8*n9 - 36*n2*n8*n9 - 28*n3*n8*n9 - 24*n4*n8*n9 - 20*n5*n8*n9 - 16*n6*n8*n9 - 16*n1*n9**2 - 8*n2*n9**2 - 8*n4*n9**2 - 4*n5*n9**2)/16.
    
    denom = n*(n-1)*(n-2)*(n-3)
    return numer / denom


def r2_Rag(G: np.ndarray, cut: bool = False, nan_as_one: bool = True) -> float:
    """
    Ragsdale and Gravel ratio: D2_Rag / Ddiag_Rag with optional clipping and nan-handling.
    """
    counts = _counts_3x3(G)
    denom = Ddiag_Rag(G, counts=counts)
    if np.isclose(denom, 0.0, atol=1e-12):
        return 1.0 if nan_as_one else np.nan
    out = D2_Rag(G, counts=counts) / denom
    return _clip01(out) if cut else float(out)


def r2_Ber(G: np.ndarray, cut: bool = False) -> float:
    """
    r2Ber estimator from the notebook (stable/private-data motivated).
    """
    n = int(G.shape[0])
    mu = G.mean(axis=0)

    aux = np.cov(G.T) / 2.0
    sig2 = np.diag(aux)
    D = aux[0, 1]

    numerator = D**2 - sig2[0] * sig2[1]
    denominator = (
        -(1.0 / n) * D**2
        + ((n - 1) / n) * sig2[0] * sig2[1]
        - ((n - 2) / (2 * (n - 1))) * D * (mu[0] - 1) * (mu[1] - 1)
    )

    if np.isclose(numerator, 0.0, atol=1e-12):
        return 1.0
    out = 1.0 + numerator / denominator
    return _clip01(out) if cut else float(out)