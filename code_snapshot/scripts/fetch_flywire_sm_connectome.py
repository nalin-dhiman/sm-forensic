#!/usr/bin/env python3
"""Fetch a signed synapse-count matrix for an Sm-like population from FlyWire.

This is best-effort scaffolding. FlyWire APIs and access requirements can change.

Prereqs:
  pip install fafbseg
  export FLYWIRE_API_TOKEN=...

Example:
  python scripts/fetch_flywire_sm_connectome.py --cell-type-regex "Sm.*" --materialization 783 --out data/sm_783.npz
"""

import sys
from pathlib import Path as _Path
sys.path.append(str(_Path(__file__).resolve().parents[1] / 'src'))

import argparse
from pathlib import Path
import os

from smforensic.connectome import save_connectome_npz
from smforensic.flywire import fetch_connectivity_signed, fetch_ids_by_type_regex


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell-type-regex", type=str, default="Sm.*", help="Regex over cell type labels")
    ap.add_argument("--materialization", type=int, default=783)
    ap.add_argument("--max-neurons", type=int, default=0, help="If >0, truncate for a quick test")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    token = os.getenv("FLYWIRE_API_TOKEN")
    if not token:
        raise SystemExit("FLYWIRE_API_TOKEN is not set")

    print(f"Querying IDs for type regex: {args.cell_type_regex}")
    ids = fetch_ids_by_type_regex(args.cell_type_regex, materialization=args.materialization)
    if args.max_neurons and len(ids) > args.max_neurons:
        ids = ids[: args.max_neurons]
    print(f"Found {len(ids)} neurons")

    print("Fetching signed connectivity (this can take a while)...")
    W = fetch_connectivity_signed(ids, ids, token=token, materialization=args.materialization)

    out = Path(args.out)
    meta = {
        "source": "FlyWire",
        "materialization": int(args.materialization),
        "cell_type_regex": args.cell_type_regex,
        "N": int(W.shape[0]),
    }
    save_connectome_npz(out, W, metadata=meta)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
