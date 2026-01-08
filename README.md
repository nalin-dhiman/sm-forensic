# Forensic Dynamical Interrogation of Serpentine Medulla Circuits  
*A conservative connectome-to-dynamics case study in the Drosophila optic lobe*

---

## Overview

This repository and submission package accompany the manuscript:

**“A forensic dynamical interrogation of Serpentine Medulla circuits in the *Drosophila* optic lobe connectome.”**

The work presents a **conservative, auditable workflow** for testing qualitative dynamical hypotheses in connectome-constrained spiking network models. Rather than optimizing high-capacity models or claiming circuit function, the study emphasizes:

- explicit modeling assumptions,
- adversarial falsification of appealing narratives,
- replication across random seeds,
- effect sizes alongside significance tests,
- negative controls, and
- robustness checks for biologically uncertain parameters.

The Serpentine Medulla (Sm) interneuron family in the *Drosophila* optic lobe is used as a **case study**, not as a claim of in vivo function.

---

## What this work does (and does not do)

### This work **does**:
- Instantiate a large connectome-derived subgraph (Sm × Sm, 4,463 neurons) as a spiking network using an adaptive exponential integrate-and-fire (AdEx) model.
- Test whether strong recurrence alone supports bistable persistent activity under strict criteria.
- Characterize oscillatory structure, external feedback signatures, and robustness to synaptic sign uncertainty.
- Provide a transparent, end-to-end pipeline that can be audited and reused.

### This work **does not**:
- Claim biological working memory, cognitive function, or validated in vivo dynamics.
- Fit cell-type-specific electrophysiology or neuromodulatory states.
- Assert equivalence between model behavior and experimental recordings.
- Optimize parameters to force desired dynamical outcomes.

Negative or null results are treated as **informative constraints**, not failures.

---

## Submission package contents

This submission package contains only the materials required to evaluate, compile, and read the manuscript:

.
├── codes/                # Simulation engine and analysis scripts
├── data/                 # Cached subgraphs and materialization IDs (no raw FlyWire data)
├── figures/              # PDF outputs referenced in manuscript
├── REPRODUCIBILITY.md    # Detailed audit trail, parameters, and assumptions





**Important:**  
The full **code snapshot** used to generate simulations and figures is provided separately as a **repository ZIP archive**. This separation is intentional, to keep the manuscript artifact lightweight and reviewer-friendly.

---

## Code availability and data access

- All simulation and analysis code is included in the accompanying repository ZIP file.
- The code is structured to regenerate all figures in the manuscript given access to the relevant connectome data.
- Raw FlyWire connectivity data are **not redistributed** here, as access may require credentials and may be subject to data-use policies.

Instead, the code:
- records FlyWire materialization identifiers and access dates,
- provides scripts to re-fetch the Serpentine Medulla subgraph when permitted,
- caches intermediate results in portable formats for reproducibility.

See `REPRODUCIBILITY.md` for details.

---

## Reproducibility philosophy

This project adopts a **forensic reproducibility stance**:

- Every major claim is evaluated across multiple random seeds.
- Effect sizes are reported alongside hypothesis tests.
- Analysis parameters (filters, PSD windows, thresholds) are fixed across conditions.
- Negative controls (e.g., linear dynamical baselines) are included to detect misleading short-horizon predictive success.
- Known biological uncertainties (e.g., neurotransmitter sign mapping) are explicitly stress-tested rather than hidden.

The goal is not to demonstrate that a model can produce an interesting phenomenon, but to test whether that phenomenon **survives attempts to falsify it**.

---

## Intended audience

This work is intended for readers interested in:

- connectome-constrained modeling,
- computational neuroscience methodology,
- spiking network dynamics,
- reproducibility and robustness in large-scale simulations.

It may be particularly useful for researchers who want a **template for cautious connectome-to-dynamics analysis**, including how to report negative results responsibly.

---

## Citation



---

## Contact

For questions about the manuscript or code:

**Correspondence:**  
`d24008@students.iitmandi.ac.in`

---

## Final note

This submission is intentionally restrained in its claims.  
Its primary contribution is methodological clarity rather than functional discovery.

Readers are encouraged to treat the results as **conditional on the stated assumptions** and to view the pipeline as a starting point for further, experimentally grounded investigation.

