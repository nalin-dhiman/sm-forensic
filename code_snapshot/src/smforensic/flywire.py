from __future__ import annotations

"""Optional FlyWire access helpers.

These utilities are *not* required to run the toy demo.

They require:
    pip install fafbseg

and a FlyWire token in the environment:
    export FLYWIRE_API_TOKEN=...

The API/behavior can change over time; treat these scripts as best-effort scaffolding.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


DEFAULT_SIGN_MAP = {
    "acetylcholine": 1,
    "gaba": -1,
    # In Drosophila, glutamate is often inhibitory via GluCl channels; this is context-dependent.
    "glutamate": -1,
    "octopamine": 1,
    "serotonin": 1,
    "dopamine": 1,
    None: 1,
}


@dataclass(frozen=True)
class FlyWireFetchConfig:
    materialization: Optional[int] = 783
    sign_map: Dict[object, int] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.sign_map is None:
            object.__setattr__(self, "sign_map", DEFAULT_SIGN_MAP)


def _require_fafbseg():
    try:
        from fafbseg import flywire  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "fafbseg is required for FlyWire access. Install with `pip install fafbseg`.\n"
            f"Original import error: {e}"
        )


def set_token(token: str) -> None:
    _require_fafbseg()
    from fafbseg import flywire
    flywire.set_chunkedgraph_secret(token, overwrite=False)


def fetch_ids_by_type_regex(cell_type_regex: str, *, materialization: Optional[int] = None) -> List[int]:
    """Fetch root IDs matching a regex over type labels (best-effort)."""
    _require_fafbseg()
    from fafbseg import flywire

    if materialization is not None:
        flywire.set_default_dataset(flywire.get_chunkedgraph(dataset=materialization))

    # This API may evolve; keep this as minimal scaffolding.
    crit = flywire.NeuronCriteria(type=cell_type_regex)
    ids = crit.get_roots()
    return [int(x) for x in ids]


def fetch_connectivity_signed(
    pre_ids: Sequence[int],
    post_ids: Sequence[int],
    *,
    token: str,
    materialization: Optional[int] = 783,
    sign_map: Optional[Dict[object, int]] = None,
) -> sp.csr_matrix:
    """Fetch a signed synapse-count matrix W[post, pre] from FlyWire.

    Signs are applied presynaptically using transmitter predictions.

    WARNING:
    - transmitter predictions have uncertainty and can be cell-type dependent.
    - a signed synapse count matrix is a simplification; true synaptic strength is unknown.
    """
    _require_fafbseg()
    from fafbseg import flywire

    set_token(token)

    if sign_map is None:
        sign_map = DEFAULT_SIGN_MAP

    # Connectivity counts
    syn = flywire.synapses.connectivity(pre_ids, post_ids)
    # syn is a DataFrame-like table with columns: pre, post, weight (counts) ...
    import pandas as pd  # local import

    df = pd.DataFrame(syn)
    if df.empty:
        return sp.csr_matrix((len(post_ids), len(pre_ids)), dtype=float)

    pre_to_j = {int(pid): j for j, pid in enumerate(pre_ids)}
    post_to_i = {int(pid): i for i, pid in enumerate(post_ids)}

    rows = df["post"].map(post_to_i).to_numpy()
    cols = df["pre"].map(pre_to_j).to_numpy()
    data = df["weight"].to_numpy(dtype=float)

    W = sp.coo_matrix((data, (rows, cols)), shape=(len(post_ids), len(pre_ids))).tocsr()

    # transmitter predictions (presynaptic)
    preds = flywire.get_transmitter_predictions(list(pre_ids), single_pred=True)
    signs = np.ones(len(pre_ids), dtype=float)
    for j, pid in enumerate(pre_ids):
        pred = preds.get(int(pid))
        nt = pred.transmitter if pred is not None else None
        signs[j] = float(sign_map.get(nt, 1))

    return W.multiply(signs[None, :]).tocsr()
