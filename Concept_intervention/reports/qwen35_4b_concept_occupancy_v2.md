# Qwen3.5-4B concept-vector J-space occupancy v2

Status: complete and numerically audited. The artifact contains 840/840
registered decompositions and 168/168 fixed-C group-bootstrap probes.

## Provenance and data quality

| Field | Value |
|---|---|
| Artifact | `artifacts/concept_intervention/qwen35_4b_concept_occupancy_v2` (438 MB) |
| Source code | `fba4738ce97b38cdea63180a17492f9cdcdc0a74` |
| Model/tokenizer | `Qwen/Qwen3.5-4B` at `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Lens SHA-256 | `cdb356a078f28f5cfc9bc2f78ac90988abb78be9b05f24624b7f7b3bb5f29446` |
| Coordinate | transformer block output / `resid_post` |
| Occupancy index SHA-256 | `467685a59142fbd014da64e8babf1746a894d477c17f3bcfdc39dba2884d8fd0` |
| Slurm | 7409717, 7409718, 7409719, 7409720; all `COMPLETED (0:0)` |

Quest compute nodes did not expose `git`, so the automatically generated
manifests recorded `git_commit=null`. This is a metadata defect, not a
numerical failure. `provenance_audit.json` records the recovered commit using
the immutable run-worktree path, login-node `rev-parse`, Slurm command paths,
config hash, and final index hash. No numerical artifact was modified.

The full numerical audit found:

- 840 finite, correctly shaped real/control error and gain curves;
- no increasing real error step and no negative coefficient;
- exact `w = w_J + w_nonJ` reconstruction for every registered stopping point;
- maximum independently recomputed error discrepancy
  \(5.6\times10^{-16}\); and
- no missing or duplicate registered combination.

## What K means

`crossing_k` is the first candidate step whose marginal reconstruction gain is
not larger than the median gain from the full-cardinality matched-norm random
dictionary. Therefore the supported sparse size is
`k_selected_before_crossing = crossing_k - 1`.

Across all 840 targets, the supported K distribution was:

| K | 0 | 1 | 2 | 3 | 4 |
|---:|---:|---:|---:|---:|---:|
| Count | 538 | 194 | 77 | 27 | 4 |

The many zeros in early layers are meaningful: a full vocabulary-sized random
dictionary can match or exceed the first J-direction's gain. They must not be
reported as successful one-atom reconstructions.

For the 84 primary positive targets, K was 0 for 38, 1 for 24, 2 for 14, 3 for
6, and 4 for 2. Supported K increases with depth. Layer 28 is the only layer
where every primary positive concept has K greater than zero.

## Registered layer-28 result

The primary intervention convention is RMSNorm-weighted. The table reports the
full-cardinality-control K, explained probe-vector energy at that K, and the
selected token J-directions.

| Concept | K | Explained at K | Selected token readout |
|---|---:|---:|---|
| admiration | 3 | 3.47% | `Excellent`, `夸`, `美中` |
| approval | 3 | 2.97% | `Ag`, `Tuttavia`, `无须` |
| curiosity | 4 | 7.40% | `答案`, `?'`, `的好奇`, `spør` |
| disapproval | 4 | 6.62% | `nor`, `除非`, `Reject`, `Зато` |
| gratitude | 2 | 1.94% | `appreciate`, `Geschenk` |
| love | 2 | 8.84% | `love`, `也喜欢` |
| optimism | 2 | 3.65% | `Hope`, `否则` |

The positive-direction semantic readout is strongest for curiosity,
disapproval, gratitude, love, optimism, and admiration. Approval remains a
registered difficult case: its token readout is mixed and its bootstrap K is
less stable. It should be retained as a negative/stress-test concept rather
than silently removed.

At layer 28, bootstrap directions have cosine 0.835–0.904 with their primary
probe directions. The full-control K varies by at most one atom for
admiration, curiosity, gratitude, love, and optimism, and by at most two for
approval and disapproval. Raw and RMSNorm-weighted conventions share most
leading token support, but RMSNorm weighting generally supports one additional
atom in the final two layers.

## Reconstruction strength and control interpretation

The sparse components are not full simulations in an L2 reconstruction sense.
At the registered K they explain only 1.94%–8.84% of the probe-vector energy.
Even at K=64, the median explained fraction over all primary positive targets
is 2.96% under raw coordinates and 4.96% under RMSNorm-weighted coordinates.

The full-cardinality random control explains much more energy at large K
(median 32.09% at K=64) because 248,077 random atoms offer strong
high-dimensional coverage. The 1/16-cardinality sensitivity control produces
larger supported K values, but it is not the primary null.

The correct conclusion is therefore:

> At layer 28, two to four token J-directions form a small, statistically
> privileged component of each primary positive concept vector. v2 does not
> show that two to four directions reconstruct the entire concept vector.

This is directly analogous to the Workspace observation that a low-variance J
component may nevertheless carry disproportionate causal effect.

## Next experiment

Proceed to an equal-magnitude causal comparison at layer 28:

- full concept direction;
- registered sparse `w_J(K)`;
- reconstruction residual `w_nonJ(K)`;
- five matched random directions;
- signed strengths around zero; and
- target and off-target concept-label probabilities plus deterministic
  generation samples.

Every direction is unit-normalized and multiplied by the same median layer-28
residual norm, so effect differences reflect direction rather than raw vector
magnitude. The intervention must be applied at `resid_post` on the final prompt
position. A successful J-component result requires a signed dose response,
agreement with the full-vector curve, and an effect larger than all matched
random controls.
