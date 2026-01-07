#!/usr/bin/env python3
"""Run the strict attractor assay on a provided connectome matrix (.npz)."""

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))

import argparse
from pathlib import Path
import numpy as np
import scipy.sparse as sp

from smforensic.config import DEFAULT_ADEX, DEFAULT_NOISE, DEFAULT_SIM, DEFAULT_SYN, SimParams
from smforensic.connectome import load_connectome_npz
from smforensic.assays import attractor_assay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--connectome", type=str, required=True, help="Path to CSR matrix saved with scipy.sparse.save_npz")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--t-on", type=float, default=500.0)
    ap.add_argument("--t-off", type=float, default=700.0)
    ap.add_argument("--amp", type=float, default=200.0, help="Pulse amplitude (pA)")
    ap.add_argument("--frac", type=float, default=0.2, help="Fraction of neurons to pulse")
    ap.add_argument("--noise-off", type=float, default=1500.0)
    ap.add_argument("--T", type=float, default=2000.0)
    args = ap.parse_args()

    con = load_connectome_npz(args.connectome)
    W = con.W
    N = W.shape[0]

    rng = np.random.default_rng(0)
    idx = rng.choice(N, size=max(1, int(args.frac * N)), replace=False)

    pulse = {"t_on_ms": args.t_on, "t_off_ms": args.t_off, "amp_pA": args.amp, "idx": idx}

    sim = SimParams(dt_ms=DEFAULT_SIM.dt_ms, T_ms=float(args.T), Ibias_pA=DEFAULT_SIM.Ibias_pA, seed=0, min_rate_hz=DEFAULT_SIM.min_rate_hz, max_rate_hz=DEFAULT_SIM.max_rate_hz)

    out = attractor_assay(
        W,
        adex=DEFAULT_ADEX,
        syn=DEFAULT_SYN,
        noise=DEFAULT_NOISE,
        sim=sim,
        seeds=args.seeds,
        pulse=pulse,
        noise_off_after_ms=float(args.noise_off),
    )

    print("Attractor assay results:")
    print(f"  N seeds: {len(args.seeds)}")
    print(f"  p-value: {out.p_value:.3g}")
    print(f"  dz: {out.effect_dz:.3g}")
    print(f"  end rate control: mean={out.end_rate_control.mean():.3f} Hz")
    print(f"  end rate pulse:   mean={out.end_rate_pulse.mean():.3f} Hz")


if __name__ == "__main__":
    main()
