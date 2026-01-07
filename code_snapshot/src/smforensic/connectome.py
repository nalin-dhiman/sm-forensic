from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class Connectome:
    """A minimal container for a connectome adjacency matrix.

    Convention
    ----------
    W is shaped (N_post, N_pre) and represents synapse counts or signed weights.
    """
    W: sp.csr_matrix
    metadata: Dict[str, object]


def save_connectome_npz(path: str | Path, W: sp.spmatrix, *, metadata: Optional[Dict[str, object]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, W.tocsr())
    if metadata is not None:
        meta_path = path.with_suffix(".meta.json")
        import json
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)


def load_connectome_npz(path: str | Path) -> Connectome:
    path = Path(path)
    W = sp.load_npz(path).tocsr()
    meta_path = path.with_suffix(".meta.json")
    metadata: Dict[str, object] = {}
    if meta_path.exists():
        import json
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    metadata.setdefault("path", str(path))
    metadata.setdefault("N", int(W.shape[0]))
    return Connectome(W=W, metadata=metadata)


def perturb_signs(W: sp.csr_matrix, frac_flip: float, *, seed: int = 0) -> sp.csr_matrix:
    """Randomly flip the sign of a fraction of nonzero edges.

    This is a crude stress test for neurotransmitter→sign uncertainty. It is not a biological model.
    """
    if not (0.0 <= frac_flip <= 1.0):
        raise ValueError("frac_flip must be in [0,1]")
    rng = np.random.default_rng(seed)
    W = W.tocsr(copy=True)
    nnz = W.nnz
    k = int(round(frac_flip * nnz))
    if k == 0:
        return W
    idx = rng.choice(nnz, size=k, replace=False)
    W.data[idx] *= -1.0
    return W


def make_mock_connectome(
    n: int = 500,
    *,
    p_in: float = 0.02,
    p_out: float = 0.002,
    frac_inhibitory: float = 0.2,
    seed: int = 0,
) -> Connectome:
    """Generate a synthetic 'Sm-like' connectome for demos.

    The goal is *not* biological realism. The goal is to ship a runnable example that exercises the pipeline.

    Structure:
    - A denser recurrent core ("in") + sparse background ("out")
    - Random synapse counts (1..5)
    - Random signs with a controllable inhibitory fraction

    Returns
    -------
    Connectome
        W is CSR.
    """
    rng = np.random.default_rng(seed)
    n_core = max(1, int(0.3 * n))
    core = np.arange(n_core)
    rest = np.arange(n_core, n)

    rows = []
    cols = []
    data = []

    def add_edges(src_idx, dst_idx, p):
        # src -> dst means col=src, row=dst
        for s in src_idx:
            mask = rng.random(len(dst_idx)) < p
            d = dst_idx[mask]
            if len(d) == 0:
                continue
            rows.extend(d.tolist())
            cols.extend([int(s)] * len(d))
            counts = rng.integers(1, 6, size=len(d))
            data.extend(counts.tolist())

    # core→core dense, rest→rest sparse, cross very sparse
    add_edges(core, core, p_in)
    add_edges(rest, rest, p_out)
    add_edges(core, rest, p_out)
    add_edges(rest, core, p_out)

    W = sp.csr_matrix((np.array(data, dtype=float), (np.array(rows), np.array(cols))), shape=(n, n))

    # Apply random signs (column-wise, like presynaptic transmitter sign)
    pre_sign = np.ones(n, dtype=float)
    inhib_idx = rng.choice(n, size=int(round(frac_inhibitory * n)), replace=False)
    pre_sign[inhib_idx] = -1.0
    W = W.multiply(pre_sign[None, :])

    meta = {
        "type": "mock",
        "N": n,
        "seed": seed,
        "p_in": p_in,
        "p_out": p_out,
        "frac_inhibitory": frac_inhibitory,
    }
    return Connectome(W=W.tocsr(), metadata=meta)
