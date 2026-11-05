"""Game-level paired bootstrap."""
import numpy as np


def paired_bootstrap(groups: np.ndarray, num_a: np.ndarray, num_b: np.ndarray, den: np.ndarray,
                     n_boot=2000, seed=0):
    """Ratio-of-sums difference (b - a) with whole-game resampling.
    groups: game id per row; num_*: numerators per row; den: denominators."""
    ug, inv = np.unique(groups, return_inverse=True)
    A = np.bincount(inv, num_a, len(ug)); B = np.bincount(inv, num_b, len(ug)); D = np.bincount(inv, den, len(ug))
    point = B.sum() / D.sum() - A.sum() / D.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ug), (n_boot, len(ug)))
    boots = B[idx].sum(1) / D[idx].sum(1) - A[idx].sum(1) / D[idx].sum(1)
    return float(point), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mean_ci(groups, x, n_boot=2000, seed=0):
    ones = np.ones_like(x, dtype=float)
    p, lo, hi = paired_bootstrap(groups, np.zeros_like(x, dtype=float), x.astype(float), ones, n_boot, seed)
    return p, lo, hi
