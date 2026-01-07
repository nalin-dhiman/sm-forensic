"""Validation gates and small analysis utilities.

These functions exist to prevent the most common failure mode in simulation-based papers:

    *a pretty figure generated from a broken run*

They are intentionally conservative and noisy: they should fail early when something looks off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import scipy.sparse as sp


def assert_nontrivial(name: str, x: np.ndarray, *, min_std: float = 1e-6, min_range: float = 1e-6) -> None:
    x = np.asarray(x)
    if x.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(x).all():
        raise ValueError(f"{name} contains non-finite values")
    if float(np.std(x)) < min_std:
        raise ValueError(f"{name} is (near) constant: std={np.std(x):.3g}")
    if float(np.max(x) - np.min(x)) < min_range:
        raise ValueError(f"{name} has (near) zero range: range={(np.max(x)-np.min(x)):.3g}")


def assert_transfer_function_ok(I_grid: np.ndarray, rates_hz: np.ndarray, *, min_max_rate: float = 5.0) -> None:
    I_grid = np.asarray(I_grid)
    rates_hz = np.asarray(rates_hz)
    if len(I_grid) != len(rates_hz):
        raise ValueError("I_grid and rates must match")
    if np.max(rates_hz) < min_max_rate:
        raise ValueError(
            f"Transfer function looks dead: max rate={np.max(rates_hz):.3g} Hz. "
            "This is often a units/config issue."
        )
    # Transfer should be nonnegative and roughly monotone in I
    if np.min(rates_hz) < -1e-6:
        raise ValueError("Transfer function has negative rates; check measurement.")
    # Not a strict monotonicity check (noise can make small violations)
    violations = np.sum(np.diff(rates_hz) < -1e-3)
    if violations > 0.2 * (len(rates_hz) - 1):
        raise ValueError(f"Transfer function is highly non-monotone ({violations} downward steps).")


@dataclass(frozen=True)
class DecompositionResult:
    t_ms: np.ndarray
    rec_drive: np.ndarray
    ext_drive: np.ndarray


def _spike_event_to_bins(t_spikes_ms: np.ndarray, dt_ms: float, T: int) -> np.ndarray:
    """Convert spike times (ms) to integer bins in [0,T)."""
    bins = np.floor(t_spikes_ms / dt_ms).astype(int)
    bins = bins[(bins >= 0) & (bins < T)]
    return bins


def estimate_sub_input_conductances(
    *,
    t_ms: np.ndarray,
    spikes_t_ms: np.ndarray,
    spikes_i: np.ndarray,
    sub_idx: np.ndarray,
    W: sp.csr_matrix,
    tau_syn_ms: float,
    dt_ms: float,
    alpha: float,
) -> DecompositionResult:
    """Decompose synaptic drive into recurrent (within-sub) and external components.

    Parameters
    ----------
    t_ms
        Time vector (ms), length T.
    spikes_t_ms, spikes_i
        Spike events. spikes_i are neuron indices (0..N-1).
    sub_idx
        Indices for the subpopulation.
    W
        Signed adjacency (post x pre).
    tau_syn_ms, dt_ms, alpha
        Must match the simulation.

    Returns
    -------
    DecompositionResult
        rec_drive and ext_drive are 1D arrays (length T) corresponding to mean signed
        synaptic current into the subpopulation from recurrent vs external sources.

    Notes
    -----
    This is a *linear* reconstruction using the same exponential kernel used in the simulator.
    It ignores dendritic nonlinearities, delays, etc.
    """
    t_ms = np.asarray(t_ms)
    T = len(t_ms)
    N = W.shape[0]
    sub_idx = np.asarray(sub_idx, dtype=int)

    if W.shape[0] != W.shape[1]:
        raise ValueError("W must be square")
    if spikes_t_ms.shape != spikes_i.shape:
        raise ValueError("spikes_t_ms and spikes_i must have same shape")
    if np.any(sub_idx < 0) or np.any(sub_idx >= N):
        raise ValueError("sub_idx out of bounds")

    # Build sparse spike count matrix S[t, neuron] as COO (events)
    bins = _spike_event_to_bins(spikes_t_ms, dt_ms, T)
    i_neur = spikes_i[: len(bins)].astype(int)
    valid = (i_neur >= 0) & (i_neur < N)
    bins = bins[valid]
    i_neur = i_neur[valid]

    data = np.ones(len(bins), dtype=float)
    S = sp.coo_matrix((data, (bins, i_neur)), shape=(T, N)).tocsr()

    # Split presynaptic sources
    mask_sub = np.zeros(N, dtype=bool)
    mask_sub[sub_idx] = True

    W_sub = W[:, mask_sub]   # all posts, sub pres
    W_ext = W[:, ~mask_sub]  # all posts, ext pres

    # Spikes from each pool
    S_sub = S[:, mask_sub]
    S_ext = S[:, ~mask_sub]

    # Filtered synaptic currents into *all* posts, then average over subpopulation posts
    decay = float(np.exp(-dt_ms / tau_syn_ms))
    I_rec = np.zeros(T, dtype=float)
    I_ext = np.zeros(T, dtype=float)

    # Precompute mapping to subpopulation posts
    post_mask = np.zeros(N, dtype=bool)
    post_mask[sub_idx] = True

    # Iterative filter: I[t] = decay*I[t-1] + alpha * W @ spikes[t-1]
    # Using matrix multiplication at each t is expensive; for reproducibility and simplicity
    # we keep it explicit. For large N you may want to move this into the simulator loop.
    I_rec_post = np.zeros(N, dtype=float)
    I_ext_post = np.zeros(N, dtype=float)

    for t in range(1, T):
        # presynaptic spikes at t-1
        sp_sub = S_sub[t - 1].toarray().ravel()
        sp_ext = S_ext[t - 1].toarray().ravel()

        I_rec_post = decay * I_rec_post + alpha * (W_sub @ sp_sub)
        I_ext_post = decay * I_ext_post + alpha * (W_ext @ sp_ext)

        I_rec[t] = float(np.mean(I_rec_post[post_mask]))
        I_ext[t] = float(np.mean(I_ext_post[post_mask]))

    return DecompositionResult(t_ms=t_ms, rec_drive=I_rec, ext_drive=I_ext)


def xcorr_peak(x: np.ndarray, y: np.ndarray, *, dt_ms: float, max_lag_ms: float = 200.0) -> Tuple[float, float]:
    """Return (peak_corr, peak_lag_ms) for the cross-correlation of x with y.

    We report the most *negative* peak (minimum correlation) because the manuscript's
    'negative feedback signature' is framed as a negative coupling.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have same shape")

    # normalize
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)

    max_lag = int(round(max_lag_ms / dt_ms))
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([np.mean(x[max_lag:-max_lag] * y[max_lag + lag : -max_lag + lag]) for lag in lags])

    # Most negative correlation
    k = int(np.argmin(corr))
    return float(corr[k]), float(lags[k] * dt_ms)
