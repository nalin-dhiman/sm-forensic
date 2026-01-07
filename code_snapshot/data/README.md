# Data formats

This repository is designed to work on *real connectomes*.

It ships with a small example file (`connectome.npz`) purely so the pipeline can run out-of-the-box.

For real analyses, you should generate your own connectome NPZ using
`code/scripts/import_connectome_edgelist.py`.

## Required NPZ keys

We intentionally use a **numeric-only** NPZ (no pickled Python objects) for portability.

### Option A: Dense matrix (OK up to a few thousand neurons)
- `W_counts`: `float` or `int` array of shape `(N, N)`
- `pre_signs`: `int8` array of shape `(N,)` with values in `{-1, 0, +1}`

### Option B: Sparse CSR stored as arrays (recommended for larger N)
- `W_data`: `float32` array
- `W_indices`: `int32` array
- `W_indptr`: `int32` array
- `W_shape`: `int` array of length 2 (e.g., `[N, N]`)
- `pre_signs`: `int8` array of shape `(N,)`

### Optional keys
- `meta_json`: JSON-encoded string containing provenance (source, units, etc.)
- `node_ids`: original node IDs (e.g., FlyWire root IDs) as `int64` of shape `(N,)`

## Edge-list CSV format (what the importer expects)

The importer expects a CSV with at least:
- `pre`: presynaptic node id (integer)
- `post`: postsynaptic node id (integer)
- `count`: synapse count or weight (numeric)

Signs can be provided in two ways:
- A nodes CSV (`--nodes`) with columns `id,sign` where sign is in `{-1,0,+1}`.
- Or an edges CSV column `sign` (per-edge sign; less common) where `sign ∈ {-1,0,+1}`.

For FlyWire, a common workflow is:
1. Export an adjacency edge list for a chosen neuron set.
2. Export neurotransmitter predictions per neuron and map them to signs.
3. Use the importer to build the numeric NPZ.

