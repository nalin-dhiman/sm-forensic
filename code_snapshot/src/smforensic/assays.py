from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from scipy.signal import welch
from scipy.ndimage import gaussian_filter1d

from .adex import AdExSimulator, SimResult
from .config import AdExParams, NoiseParams, SimParams, SynapseParams
from .stats import cohens_dz, paired_wilcoxon
from .validate import assert_nontrivial, assert_transfer_function_ok


def population_rate_cv(rate_hz: np.ndarray, *, dt_ms: float, smooth_sigma_ms: float = 5.0) -> float:
    """A simple synchrony proxy: CV of the smoothed population rate."""
    rate_hz = np.asarray(rate_hz, dtype=float)
    sigma_steps = max(1, int(round(smooth_sigma_ms / dt_ms)))
    r = gaussian_filter1d(rate_hz, sigma=sigma_steps)
    m = float(np.mean(r))
    s = float(np.std(r))
    if m == 0:
        return 0.0
    return s / m


def gamma_band_power(rate_hz: np.ndarray, *, dt_ms: float, band_hz: Tuple[float, float] = (30.0, 80.0), nperseg: int = 2048) -> Tuple[np.ndarray, np.ndarray, float]:
    """Welch PSD + integrated power in a frequency band."""
    rate_hz = np.asarray(rate_hz, dtype=float)
    fs = 1000.0 / dt_ms  # Hz (since dt_ms)
    f, Pxx = welch(rate_hz, fs=fs, nperseg=min(nperseg, len(rate_hz)))
    lo, hi = band_hz
    mask = (f >= lo) & (f <= hi)
    power = float(np.trapz(Pxx[mask], f[mask])) if np.any(mask) else 0.0
    return f, Pxx, power


def estimate_transfer_curve(
    I_grid_pA: np.ndarray,
    *,
    adex: AdExParams,
    syn: SynapseParams,
    sim: SimParams,
    T_ms: float = 1000.0,
    burn_in_ms: float = 200.0,
) -> Dict[str, np.ndarray]:
    """Empirical AdEx transfer curve: steady-state rate vs constant current.

    Uses a single neuron with no recurrent inputs and no noise.
    """

    I_grid_pA = np.asarray(I_grid_pA, dtype=float)
    rates = np.zeros_like(I_grid_pA, dtype=float)

    W0 = sp.csr_matrix((1, 1), dtype=float)
    noise0 = NoiseParams(tau_ou_ms=10.0, sigma_ou_pA=0.0)

    for k, I in enumerate(I_grid_pA):
        sim_k = SimParams(dt_ms=sim.dt_ms, T_ms=float(T_ms), Ibias_pA=float(I), seed=sim.seed, min_rate_hz=sim.min_rate_hz, max_rate_hz=sim.max_rate_hz)
        simr = AdExSimulator(W0, adex=adex, syn=syn, noise=noise0, sim=sim_k)
        res = simr.run(T_ms=float(T_ms), record_spikes=True)
        # steady-state rate: last (T - burn_in) window
        burn = int(round(burn_in_ms / sim.dt_ms))
        rates[k] = float(np.mean(res.rate_hz[burn:]))

    assert_transfer_function_ok(I_grid_pA, rates)
    return {"I_pA": I_grid_pA, "rate_hz": rates}


@dataclass(frozen=True)
class AttractorAssayResult:
    end_rate_control: np.ndarray
    end_rate_pulse: np.ndarray
    p_value: float
    effect_dz: float


def attractor_assay(
    W: sp.csr_matrix,
    *,
    adex: AdExParams,
    syn: SynapseParams,
    noise: NoiseParams,
    sim: SimParams,
    seeds: Sequence[int],
    pulse: Dict[str, object],
    noise_off_after_ms: float = 1500.0,
    end_window_ms: float = 250.0,
) -> AttractorAssayResult:
    """Strict bistability assay: compare end-of-trial rates for control vs pulse."""
    end_rates_control = []
    end_rates_pulse = []
    dt = sim.dt_ms
    end_steps = int(round(end_window_ms / dt))

    for s in seeds:
        sim_s = SimParams(dt_ms=sim.dt_ms, T_ms=sim.T_ms, Ibias_pA=sim.Ibias_pA, seed=int(s), min_rate_hz=sim.min_rate_hz, max_rate_hz=sim.max_rate_hz)
        simr = AdExSimulator(W, adex=adex, syn=syn, noise=noise, sim=sim_s)
        # Control
        res_c = simr.run(noise_off_after_ms=noise_off_after_ms, pulse=None, record_spikes=False)
        # Pulse
        simr.reset(seed=int(s))
        res_p = simr.run(noise_off_after_ms=noise_off_after_ms, pulse=pulse, record_spikes=False)

        end_rates_control.append(float(np.mean(res_c.rate_hz[-end_steps:])))
        end_rates_pulse.append(float(np.mean(res_p.rate_hz[-end_steps:])))

    x = np.array(end_rates_pulse)
    y = np.array(end_rates_control)
    _, p = paired_wilcoxon(x, y)
    dz = cohens_dz(x, y)

    return AttractorAssayResult(end_rate_control=y, end_rate_pulse=x, p_value=float(p), effect_dz=float(dz))
