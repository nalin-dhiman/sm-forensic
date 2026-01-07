"""Configuration defaults.

This repo intentionally keeps configuration explicit and editable.

Important:
- Do NOT commit FlyWire tokens. Use the environment variable `FLYWIRE_API_TOKEN`.
- The defaults below mirror the parameter table in the accompanying manuscript.
  They are not claimed to be a biophysically fitted model of any specific Sm cell type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import os


@dataclass(frozen=True)
class AdExParams:
    # Core membrane + spike mechanism
    C_pF: float = 281.0
    gL_nS: float = 30.0
    EL_mV: float = -70.6
    VT_mV: float = -50.4
    DeltaT_mV: float = 2.0
    Vreset_mV: float = -70.6
    Vspike_mV: float = 0.0

    # Adaptation
    tau_w_ms: float = 144.0
    a_nS: float = 4.0
    b_pA: float = 0.02  # Small in the original draft; keep explicit.

    # Refractory
    tau_ref_ms: float = 2.0


@dataclass(frozen=True)
class SynapseParams:
    tau_syn_ms: float = 5.0
    alpha_pA_per_count: float = 6.0  # global synapse-count → current scale ("α")


@dataclass(frozen=True)
class NoiseParams:
    # Ornstein–Uhlenbeck injected current noise
    tau_ou_ms: float = 10.0
    sigma_ou_pA: float = 100.0


@dataclass(frozen=True)
class SimParams:
    dt_ms: float = 0.1
    T_ms: float = 2000.0

    # Constant bias current added to every neuron
    Ibias_pA: float = 700.0

    # Deterministic seed used for *any* RNG (numpy + python)
    seed: int = 0

    # Sanity bounds (used by validation gates)
    min_rate_hz: float = 0.01
    max_rate_hz: float = 300.0


# FlyWire (optional) ---------------------------------------------------------

FLYWIRE_API_TOKEN: Optional[str] = os.getenv("FLYWIRE_API_TOKEN")
"""Set this in your environment if you want to fetch real FlyWire connectivity."""


# Convenience bundles --------------------------------------------------------

DEFAULT_ADEX = AdExParams()
DEFAULT_SYN = SynapseParams()
DEFAULT_NOISE = NoiseParams()
DEFAULT_SIM = SimParams()
