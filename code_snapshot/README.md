# sm-forensic

A small, auditable toolbox for **testing** (and often *falsifying*) connectome-to-dynamics hypotheses under explicit, stress‑tested assumptions.


> Dense wiring does not automatically imply a specific dynamical “function”.  
> If you want to claim bistability / oscillators / memory, you should be able to **fail to kill** that claim under adversarial checks.

What this is *not*:
- not a fitted biophysical model of the Drosophila optic lobe,
- not an attempt to “predict function from wiring”,
- not a substitute for experimental perturbations.

## Repository structure

- `src/smforensic/` — lightweight simulator + assays + validation gates
- `scripts/` — runnable entry points (toy demo + optional FlyWire fetch scaffolding)
- `paper/` —figures used in the draft
- `data/` — data files
- `outputs/` — where scripts write cached results

## Quickstart (toy demo)

```bash
pip install -r requirements.txt
pip install -e .
python scripts/demo_mock_pipeline.py
```

This runs the pipeline on a **synthetic** “Sm‑like” graph to verify that the code path works.

## Using real FlyWire data (optional)

You need:
- FlyWire access
- a FlyWire token exported as `FLYWIRE_API_TOKEN`
- the Python package `fafbseg`

Example (best-effort scaffolding; APIs can change):

```bash
pip install -r requirements.txt
pip install fafbseg
export FLYWIRE_API_TOKEN="..."
python scripts/fetch_flywire_sm_connectome.py --out data/sm_flywire_783.npz
```

## Reproducibility philosophy

The code tries to be “forensic” in a boring way:
- deterministic seeding,
- sanity checks that fail early,
- explicit reporting of effect sizes,
- stress tests around uncertain assumptions (e.g., synapse sign mapping).

If you do *not* like a modeling assumption, the intent is that you can change it in one place and re-run.

## License

MIT
