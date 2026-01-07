from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from .config import AdExParams, NoiseParams, SimParams, SynapseParams
from .utils import seed_everything


@dataclass
class SimResult:
    t_ms: np.ndarray
    rate_hz: np.ndarray
    spikes_t_ms: np.ndarray
    spikes_i: np.ndarray
    meta: Dict[str, object]


class AdExSimulator:
    """Current-based AdEx network simulator with exponential synapses and OU noise.

    Design goals:
    - Clear equations and explicit units (mV, ms, pA, pF, nS).
    - Deterministic by default (seeded).
    - Reasonably efficient for medium-sized networks when W is sparse.
    - Easy to modify for stress tests (sign flips, alpha sweeps, noise-off, pulses).

    This is not a highly-optimized simulator. If you want speed at scale, use Brian2 or NEST.
    """

    def __init__(
        self,
        W: sp.csr_matrix,
        *,
        adex: AdExParams,
        syn: SynapseParams,
        noise: NoiseParams,
        sim: SimParams,
    ) -> None:
        if W.shape[0] != W.shape[1]:
            raise ValueError("W must be square (post x pre)")
        self.W = W.tocsr()
        self.N = int(W.shape[0])
        self.adex = adex
        self.syn = syn
        self.noise = noise
        self.sim = sim

        # Precompute decays
        self._decay_syn = float(np.exp(-sim.dt_ms / syn.tau_syn_ms))
        self._decay_ou = float(np.exp(-sim.dt_ms / noise.tau_ou_ms))

        # OU exact update scale factor
        self._ou_sigma_step = float(noise.sigma_ou_pA * np.sqrt(1.0 - np.exp(-2.0 * sim.dt_ms / noise.tau_ou_ms)))

        # State (allocated on reset)
        self.v = np.empty(self.N, dtype=np.float32)
        self.w = np.empty(self.N, dtype=np.float32)
        self.I_syn = np.empty(self.N, dtype=np.float32)
        self.eta = np.empty(self.N, dtype=np.float32)
        self.refrac = np.zeros(self.N, dtype=np.int32)

        self.rng = seed_everything(sim.seed)
        self.reset()

    def reset(self, *, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.rng = seed_everything(int(seed))
        self.v.fill(self.adex.EL_mV)
        self.w.fill(0.0)
        self.I_syn.fill(0.0)
        self.eta.fill(0.0)
        self.refrac.fill(0)

    def run(
        self,
        *,
        T_ms: Optional[float] = None,
        noise_off_after_ms: Optional[float] = None,
        pulse: Optional[Dict[str, object]] = None,
        record_spikes: bool = True,
    ) -> SimResult:
        """Run a simulation.

        Parameters
        ----------
        T_ms
            Duration in ms. Defaults to sim.T_ms.
        noise_off_after_ms
            If provided, sets OU sigma to zero after this time (deterministic tail).
        pulse
            Optional dict with keys:
                - "t_on_ms": float
                - "t_off_ms": float
                - "amp_pA": float
                - "idx": 1D array-like of neuron indices (subset)
        record_spikes
            If True, store spike events.

        Returns
        -------
        SimResult
        """
        dt = float(self.sim.dt_ms)
        T_ms = float(self.sim.T_ms if T_ms is None else T_ms)
        steps = int(np.round(T_ms / dt))
        t_ms = np.arange(steps, dtype=np.float32) * dt

        rate_hz = np.zeros(steps, dtype=np.float32)

        # Spike event buffers (python lists, then concatenate)
        spikes_t = []
        spikes_i = []

        # Previous-step spikes as float vector for sparse matmul
        s_prev = np.zeros(self.N, dtype=np.float32)

        # Refractory in steps
        ref_steps = int(np.round(self.adex.tau_ref_ms / dt))

        # Pulse config
        if pulse is not None:
            idx = np.asarray(pulse.get("idx"), dtype=int)
            t_on = float(pulse.get("t_on_ms", 0.0))
            t_off = float(pulse.get("t_off_ms", 0.0))
            amp = float(pulse.get("amp_pA", 0.0))
        else:
            idx = np.array([], dtype=int)
            t_on = t_off = amp = 0.0

        for k, t in enumerate(t_ms):
            # Determine OU sigma at this time
            if noise_off_after_ms is not None and t >= float(noise_off_after_ms):
                ou_sigma_step = 0.0
            else:
                ou_sigma_step = self._ou_sigma_step

            # OU update (exact discretization)
            if ou_sigma_step == 0.0:
                self.eta *= self._decay_ou
            else:
                self.eta = self.eta * self._decay_ou + ou_sigma_step * self.rng.standard_normal(self.N).astype(np.float32)

            # Synaptic current update
            if self.W.nnz > 0:
                kick = self.syn.alpha_pA_per_count * (self.W @ s_prev)  # (N,)
            else:
                kick = 0.0
            self.I_syn = self.I_syn * self._decay_syn + kick

            # External current (bias + noise)
            I_ext = self.sim.Ibias_pA + self.eta

            # Optional pulse current (additional injected current)
            if pulse is not None and (t_on <= t < t_off) and idx.size > 0 and amp != 0.0:
                I_ext = I_ext.copy()
                I_ext[idx] += amp

            # AdEx dynamics
            # C dv/dt = -gL(v-EL) + gL*DeltaT*exp((v-VT)/DeltaT) - w + I_ext + I_syn
            exp_term = self.adex.gL_nS * self.adex.DeltaT_mV * np.exp((self.v - self.adex.VT_mV) / self.adex.DeltaT_mV)
            dv = (-self.adex.gL_nS * (self.v - self.adex.EL_mV) + exp_term - self.w + I_ext + self.I_syn) / self.adex.C_pF
            dw = (self.adex.a_nS * (self.v - self.adex.EL_mV) - self.w) / self.adex.tau_w_ms

            # Apply refractory: clamp v, but still allow w to relax
            active = self.refrac <= 0
            self.v[active] = self.v[active] + dv[active] * dt
            self.v[~active] = self.adex.Vreset_mV
            self.w = self.w + dw * dt

            # Spike + reset
            spk = (self.v >= self.adex.Vspike_mV) & active
            if np.any(spk):
                if record_spikes:
                    ii = np.flatnonzero(spk)
                    spikes_i.append(ii.astype(np.int32))
                    spikes_t.append(np.full(ii.shape, t, dtype=np.float32))
                self.v[spk] = self.adex.Vreset_mV
                self.w[spk] += self.adex.b_pA
                self.refrac[spk] = ref_steps

            # decrement refractory counters
            self.refrac = np.maximum(self.refrac - 1, 0)

            # population rate in Hz (spikes / (N * dt_s))
            rate_hz[k] = float(np.sum(spk)) / (self.N * (dt * 1e-3))

            # update s_prev for next step (float vector)
            s_prev[:] = 0.0
            if np.any(spk):
                s_prev[spk] = 1.0

        if record_spikes and len(spikes_i) > 0:
            spikes_i_arr = np.concatenate(spikes_i)
            spikes_t_arr = np.concatenate(spikes_t)
        else:
            spikes_i_arr = np.array([], dtype=np.int32)
            spikes_t_arr = np.array([], dtype=np.float32)

        meta = {
            "N": self.N,
            "dt_ms": dt,
            "T_ms": T_ms,
            "seed": self.sim.seed,
            "noise_off_after_ms": noise_off_after_ms,
            "pulse": pulse or None,
        }

        return SimResult(t_ms=t_ms, rate_hz=rate_hz, spikes_t_ms=spikes_t_arr, spikes_i=spikes_i_arr, meta=meta)

