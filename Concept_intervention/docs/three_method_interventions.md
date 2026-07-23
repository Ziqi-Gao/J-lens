# Three-method intervention protocol

## Stable experiment identities

This protocol uses stable, unversioned experiment names because later analyses
will cite these artifact roots directly:

| Scope | Experiment name and artifact root |
| --- | --- |
| Shared inputs | `shared_intervention_protocol` |
| J-component | `j_component_intervention` |
| ITI | `iti_intervention` |
| RAPTOR | `raptor_intervention` |
| Final comparison | `intervention_comparison` |

Earlier versioned runs and YAML files are historical artifacts and are not
overwritten.

## Common model, data, rows, layers, and prompts

All three methods pin `Qwen/Qwen3.5-4B` and its tokenizer at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. They use the same GoEmotions
seven-concept artifact at upstream revision
`add492243ff905527e67aeb8b80c082af02207c3`, the same chat-template/BOS policy,
and the same generation settings.

`shared_intervention_protocol/selection/row_manifest.json` records an exact
positive/negative row balance for each concept and split. Selection is
deterministic, keeps whole groups, and fails if a group crosses labels or
train/validation/test. The shared residual activation coordinate is the final
non-padding token at block output (`resid_post`).

RAPTOR-style StandardScaler plus L2-LBFGS probes are fit at layers
`[3,7,11,15,19,23,27]`. Their 100-value logarithmic C grid spans
`[1e-4,100]`; C is chosen using grouped CV inside train only. Validation
accuracy selects each concept's Top-6, with smaller layer ID breaking a tie.
The fixed C is then refit on train+validation. Test metrics are reported but do
not choose C or layers. The authoritative layer set is
`shared_intervention_protocol/selection/layer_selection.json`.

The 28 candidate-label prompts use `choose_*` and `complete_*` for validation
selection and `classify_*` and `report_*` for held-out direct evaluation. An
additional frozen bank has 16 validation and 16 test neutral open-ended
prompts. Every grid point saves greedy generation plus samples with seeds
`[1001,2002,3003]`, 128 new tokens, temperature 0.75, top-p 0.95, repetition
penalty 1.1, and no-repeat-ngram size 3.

## Independent method implementations

### J-component

The official Jacobian-lens implementation is pinned at
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. It refits float32 maps from every
candidate source layer to target layer 31 using the same 1,000 prompt IDs.

Each layer's newly fitted shared concept vector is decomposed over
RMSNorm-weighted unit J-token directions with standard non-negative Gradient
Pursuit. Primary plus four group-bootstrap probes are compared with five
same-size random dictionaries through K=64. `K_measured` is the median support
before the first random crossing; `K_used=max(1,K_measured)`. K=0 floors and
K=64 right-censoring remain explicit.

Full, J, non-J, and five random conditions use the selected six layers. Each
layer receives its full signed strength times that layer's mean development
residual norm; there is no division by the square root of the layer count.

### ITI

The attributed MIT-licensed core comes from
`likenneth/honest_llama@2c6b2179be7b5aa8f0a171688cf9e01b812ca327`.
Per-head logistic probes, positive-minus-negative mass means,
projection-standard-deviation scaling, and last-token pre-`o_proj` addition
retain the author implementation.

`ITI-native` uses the author's global accuracy ranking. `ITI-layer-matched`
places the best head from each shared selected layer first and then resumes the
global ranking, ensuring all six layers are represented. Probe-weight and five
original-style random controls are sensitivities.

### RAPTOR

RAPTOR is used from an external checkout pinned at
`Ziqi-Gao/RAPTOR@cf7405899174af39f3970e093e4b86bf0972ff87`. The upstream root
does not provide a license, so its source is neither copied nor vendored.
Preflight verifies the checkout commit. Shared probe fitting directly calls
the author's `raptor.probes.tuning.tune_raptor_c` inside every group-safe
training fold, then aggregates fold accuracies without exposing the registered
validation split. Runtime directly calls the author's
`register_steering_hooks_sequential` and `compute_adaptive_epsilon`. The local
adapter only exposes Qwen3.5's nested block list in the layout expected by
upstream code and records before/after telemetry. The test-accuracy filter and
position coefficients are disabled. The pinned `singlelr` loader discards its
stored per-layer intercepts, and the author's steering CLI uses the global
default `bias=0.0`; the experiment preserves that behavior and records it in
provenance.

## Outputs and comparison

Generation is sharded by concept and scientific grid point:

- J-component: 56 shards per concept;
- RAPTOR: 9 shards per concept;
- ITI: 441 shards per concept.

Each shard contains candidate scores, full generations, method-blind
generations, a private blind-ID map, token IDs, text, per-token log
probabilities, direction/head/K metadata, and per-forward injection telemetry.
Method indexes reject missing or duplicated grid indices, inconsistent layer
sets, and inconsistent zero-strength outputs.

No LLM-as-judge is called. Direct outputs are target log probability,
candidate-normalized probability, target margin, rank, and off-target
probabilities. Hyperparameters are selected on validation prompt families, and
the final comparison reports paired held-out target-margin change relative to
each method's own zero/no-hook baseline. Blind exports are retained for a
separately registered future judge.

## Slurm completion contract

Run:

```bash
Concept_intervention/scripts/submit_three_method_interventions.sh
```

The dependency graph is:

```text
preflight
|-- J-lens refit
`-- residual capture
    |-- shared layer selection -> bootstrap probes
    |   |-- J occupancy
    |   |-- RAPTOR smoke
    |   `-- ITI fit (after head capture)
    `-- ITI head capture

J/ITI/RAPTOR smoke -> hash/identity smoke gate
                     -> submit full grid arrays -> method indexes -> comparison index
```

An `afterok` CPU gate first validates the registered smoke conditions and all
score/generation hashes, writes `intervention_comparison/smoke_gate.json`, and
only then submits the exhaustive arrays. Thus the full jobs are not submitted
before the GPU smokes have actually passed. The experiment is complete only when
`artifacts/concept_intervention/intervention_comparison/index.json` exists and
contains `"complete": true`. Pending jobs, launchers, or partial method
directories are not experimental results.
