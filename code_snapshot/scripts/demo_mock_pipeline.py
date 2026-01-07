#!/usr/bin/env python3
"""Toy demo: run a minimal pipeline on a synthetic connectome.

This is meant to:
- verify that installation works,
- exercise the main code paths,
- avoid any FlyWire dependency.
"""

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))

from pathlib import Path
import numpy as np

from smforensic.config import DEFAULT_ADEX, DEFAULT_NOISE, DEFAULT_SIM, DEFAULT_SYN, SimParams
from smforensic.connectome import make_mock_connectome, save_connectome_npz
from smforensic.adex import AdExSimulator
from smforensic.assays import attractor_assay


def main() -> None:
    out_dir = Path("outputs/demo")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build mock connectome
    con = make_mock_connectome(n=300, seed=0)
    save_connectome_npz(out_dir / "mock_connectome.npz", con.W, metadata=con.metadata)
    W = con.W

    # 2) Run one simulation
    sim = SimParams(dt_ms=0.1, T_ms=1000.0, Ibias_pA=700.0, seed=0, min_rate_hz=0.01, max_rate_hz=300.0)
    simr = AdExSimulator(W, adex=DEFAULT_ADEX, syn=DEFAULT_SYN, noise=DEFAULT_NOISE, sim=sim)
    res = simr.run(record_spikes=False)

    print(f"Demo run mean rate: {np.mean(res.rate_hz):.2f} Hz")

    # 3) Run a strict attractor assay (pulse on random 20% of neurons)
    rng = np.random.default_rng(0)
    idx = rng.choice(W.shape[0], size=int(0.2 * W.shape[0]), replace=False)

    pulse = {"t_on_ms": 300.0, "t_off_ms": 500.0, "amp_pA": 200.0, "idx": idx}
    assay = attractor_assay(
        W,
        adex=DEFAULT_ADEX,
        syn=DEFAULT_SYN,
        noise=DEFAULT_NOISE,
        sim=SimParams(dt_ms=0.1, T_ms=1000.0, Ibias_pA=700.0, seed=0, min_rate_hz=0.01, max_rate_hz=300.0),
        seeds=[0, 1, 2, 3, 4],
        pulse=pulse,
        noise_off_after_ms=700.0,
        end_window_ms=200.0,
    )

    print("Attractor assay (toy demo, not biological):")
    print(f"  p-value: {assay.p_value:.3g}")
    print(f"  dz: {assay.effect_dz:.3g}")
    print(f"  mean end rate control: {assay.end_rate_control.mean():.2f} Hz")
    print(f"  mean end rate pulse:   {assay.end_rate_pulse.mean():.2f} Hz")


if __name__ == "__main__":
    main()
