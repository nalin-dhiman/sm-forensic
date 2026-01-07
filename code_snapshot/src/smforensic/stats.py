from __future__ import annotations

from typing import Callable, Tuple
import numpy as np
from scipy import stats


def cohens_dz(x: np.ndarray, y: np.ndarray) -> float:
    """Within-subject effect size dz for paired samples."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    sd = np.std(d, ddof=1)
    if sd == 0:
        return 0.0
    return float(np.mean(d) / sd)


def paired_wilcoxon(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Paired Wilcoxon signed-rank test (two-sided). Returns (stat, p)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    stat, p = stats.wilcoxon(x, y, zero_method="wilcox", correction=False, alternative="two-sided")
    return float(stat), float(p)


def bootstrap_ci_paired(
    x: np.ndarray,
    y: np.ndarray,
    func: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """Bootstrap CI for a statistic of paired differences.

    Returns (lo, hi) for the central CI interval.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    d = x - y
    n = len(d)
    if n == 0:
        raise ValueError("Empty sample")
    stats_boot = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats_boot[i] = func(d[idx])
    alpha = 1 - ci
    lo = np.quantile(stats_boot, alpha / 2)
    hi = np.quantile(stats_boot, 1 - alpha / 2)
    return float(lo), float(hi)
