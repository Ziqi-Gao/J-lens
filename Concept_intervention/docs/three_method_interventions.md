# Three-method intervention protocol

## Immutable experiment identity

The registered parent experiment is
`qwen35_4b_three_method_intervention_v1`. The three methods retain independent
child roots so their artifacts can be maintained and cited separately:

| Scope | Child artifact root |
| --- | --- |
| Shared inputs | `shared_intervention_protocol` |
| J-component | `j_component_intervention` |
| ITI | `iti_intervention` |
| RAPTOR | `raptor_intervention` |
| Final comparison | `intervention_comparison` |

Every child is located below
`artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/`.
The Slurm environment loads only:

- `qwen35_4b_three_method_intervention_v1_shared.yaml`;
- `qwen35_4b_three_method_intervention_v1_j_component.yaml`;
- `qwen35_4b_three_method_intervention_v1_iti.yaml`;
- `qwen35_4b_three_method_intervention_v1_raptor.yaml`.

The four earlier unversioned scientific YAMLs and their artifact roots are
preserved unchanged as historical experiments and are never consulted by this
DAG. Grid manifests are immutable: a pre-existing shard without the exact same
commit, config, model/data, prompt, row, and layer identity is rejected rather
than reused.

## Common model, data, rows, layers, and prompts

All three methods pin `Qwen/Qwen3.5-4B` and its tokenizer at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. They use the same GoEmotions
seven-concept artifact at upstream revision
`add492243ff905527e67aeb8b80c082af02207c3`, the same chat-template/BOS policy,
and the same generation settings.

The child path `shared_intervention_protocol/selection/row_manifest.json`
records an exact positive/negative row balance for each concept and split.
Selection is deterministic, keeps whole groups, and fails if a group crosses
labels or train/validation/test. The shared residual activation coordinate is
the final non-padding token at block output (`resid_post`).

RAPTOR-style StandardScaler plus L2-LBFGS probes are fit at layers
`[3,7,11,15,19,23,27]`. Their 100-value logarithmic C grid spans
`[1e-4,100]`; C is chosen using grouped CV inside train only. Validation
accuracy selects each concept's Top-6, with smaller layer ID breaking a tie.
The fixed C is then refit on train+validation. Test metrics are reported but do
not choose C or layers. The authoritative layer set is
`shared_intervention_protocol/selection/layer_selection.json`.

The 28 candidate-label prompts contain exactly the frozen seven label IDs.
They use `choose_*` and `complete_*` for validation selection and `classify_*`
and `report_*` for held-out direct evaluation; the method indexes reject any
missing/extra label or family/split mismatch. An additional frozen bank has 16
validation and 16 test neutral open-ended prompts. Every grid point saves
greedy generation plus samples with seeds
`[1001,2002,3003]`, 128 new tokens, temperature 0.75, top-p 0.95, repetition
penalty 1.1, and no-repeat-ngram size 3. The prompt-bank content hashes,
ordered prompt-ID hash, expected prompt count, and expected row count are
frozen in the run manifest and checked against every shard's generation
contract.

## Independent method implementations

### J-component

The official Jacobian-lens implementation is pinned at
`581d398613e5602a5af361e1c34d3a92ea82ba8e`. It refits float32 maps from every
candidate source layer to target layer 31 using the same 1,000 prompt IDs.
Preflight reads the installed package's PEP 610 provenance and requires its
observed VCS commit to equal this pin.

Each layer's newly fitted shared concept vector is decomposed over
RMSNorm-weighted unit J-token directions with standard non-negative Gradient
Pursuit. The one-dimensional gradient step is projected to non-negative
coefficients, then exactly line-searched on the resulting feasible segment so
projection cannot increase reconstruction error; this is not an
active-support NNLS refit. Primary plus four group-bootstrap probes are
compared with five same-size random dictionaries through K=64. `K_measured` is
the median support before the first random crossing;
`K_used=max(1,K_measured)`. K=0 floors and K=64 right-censoring remain
explicit.

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
original-style random controls are sensitivities. Native random controls keep
the author's seeded global permutation exactly. Layer-matched random controls
start from that same seeded permutation, move its first encountered head from
each selected layer to the prefix while preserving encounter order, and then
resume the remaining permutation. Thus the K=8 layer-matched controls cover all
six layers without changing the native control definition.

### RAPTOR

RAPTOR is used from an external checkout pinned at
`Ziqi-Gao/RAPTOR@cf7405899174af39f3970e093e4b86bf0972ff87`. The upstream root
does not provide a license, so its source is neither copied nor vendored.
Preflight verifies the checkout commit and directly checks the pinned
`compute_adaptive_epsilon` on steering and no-steering cases against the
registered formula. Shared probe fitting directly calls
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
Telemetry must contain exactly one event for every generated-token × selected-
layer pair. It includes the generation/forward step, layer, sequence shape,
per-event injected norm, per-layer accumulated norm, and explicit total
injected norm; J additionally records strength/residual norm/kind, ITI records
the multiplier, and RAPTOR records target probability, pre-intervention
probability/logit, epsilon, and whether steering occurred. Method indexes
reject missing or duplicated grid
indices, wrong grid conditions, empty or cardinality-mismatched
candidate/generation files, duplicate IDs, incomplete token/logprob records,
malformed telemetry, mixed prompt contracts or provenance, inconsistent layer
sets, and non-identical zero-strength token IDs, text, or per-token log
probabilities. ITI direction metrics contain a recursive SHA-256 registry for
every `.npz`; loading, method indexing, and final comparison each revalidate
the registered bytes. Each method index seals every target summary, shard summary,
candidate score, generation, and blind-export hash. The final comparison
revalidates those seals before reading any score and additionally requires the
J full/zero, RAPTOR no-hook, and ITI native/zero outputs and candidate log
probabilities to agree.

No LLM-as-judge is called. Direct outputs are target log probability,
candidate-normalized probability, target margin, rank, and off-target
probabilities. Hyperparameters are selected on validation prompt families, and
the final comparison reports paired held-out target-margin change relative to
each method's own zero/no-hook baseline. Blind exports are retained for a
separately registered future judge.

## Clean-server bootstrap

No artifact from an earlier server is required for a from-scratch run. Clone
the published experiment branch, create an environment from the lockfile, and
choose a persistent run root with enough space for model cache, activations,
head captures, generations, and logs:

```bash
git clone --branch codex/three-method-interventions \
  git@github.com:Ziqi-Gao/J-lens.git J-lens
cd J-lens
uv sync --extra dev --extra llm

export CODE_ROOT="$PWD"
export RUN_ROOT=/absolute/persistent/path/jlens-three-method
export JLENS_PYTHON="$CODE_ROOT/.venv/bin/python"
export HF_HOME="$RUN_ROOT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME"
```

The bootstrap command runs online on a login/download node. It downloads the
model and tokenizer at `851bf6e...`, downloads the three GoEmotions parquet
files at `add4922...`, verifies their checked-in SHA-256 allowlist, prepares the
deterministic seven-concept dataset and fit prompts, checks out RAPTOR at
`cf740589...`, and then runs the same offline config/data/source preflight used
by Slurm:

```bash
Concept_intervention/scripts/bootstrap_three_method_server.sh
```

Scientific batch jobs are offline. The bootstrap must finish with
`bootstrap_complete=true` before submission. The official Jacobian-lens
package at `581d398...` is installed by the `llm` extra and its observed PEP
610 VCS commit is checked by preflight.

Set the scheduler mapping without editing any scientific YAML or Slurm file.
The values below are examples; use the new cluster's actual names. The wrapper
passes these options on every initial, smoke, full-grid, and index submission,
including submissions made later by the smoke gate:

```bash
export SLURM_ACCOUNT=my_account
export SLURM_CPU_PARTITION=cpu
export SLURM_GPU_PARTITION=gpu
export SLURM_GPU_GRES=gpu:1          # for example gpu:a100:1 or gpu:h100:1
unset SLURM_EXCLUDE                  # or export a real comma-separated node list
export SLURM_BOOTSTRAP_LIMIT=7
export SLURM_OCCUPANCY_LIMIT=4
export SLURM_J_LIMIT=4
export SLURM_RAPTOR_LIMIT=4
export SLURM_ITI_LIMIT=4

Concept_intervention/scripts/submit_three_method_interventions.sh \
  | tee "$RUN_ROOT/three_method_submission.txt"
```

The checked-in resource envelope is one GPU and 64 GB host RAM for GPU jobs;
CPU stages reach 16 cores and 128 GB RAM. J-lens fitting requests 24 hours,
capture/occupancy request 12 hours, and generation shards request 8 hours.
Default array throttles are 10 occupancy, 16 J/RAPTOR generation, and 24 ITI
generation tasks. Set the corresponding `SLURM_*_LIMIT` variables to the new
cluster's per-user limits; changing launchers or scientific configs is not
necessary.

Always launch this registered DAG through
`submit_three_method_interventions.sh`. Do not directly run `sbatch` on one of
the component `.slurm` files on a different cluster: their checked-in
`#SBATCH` headers document the original Quest resource envelope, while the
submit wrapper supplies the new cluster's account, partition, GPU type, and
array limits as command-line overrides. Both `CODE_ROOT` and `RUN_ROOT` must be
visible at the same absolute paths on every compute node.

Use only a clean, immutable Git checkout. The submitter records the exact
commit and every compute job rejects a checkout that moves afterward. Keep
`CODE_ROOT`, `RUN_ROOT`, `JLENS_PYTHON`, Hugging Face cache variables, and the
`SLURM_*` mapping/throttle variables exported until the initial DAG has been
submitted. After that, Slurm carries them through the full dependency graph.

## Slurm completion contract

After completing the clean-server bootstrap, run:

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

An `afterok` CPU gate first validates the registered smoke conditions and the
full prompt-by-decoding generation contract (not only file hashes), writes
`intervention_comparison/smoke_gate.json`, and only then submits the exhaustive
arrays. Thus the full jobs are not submitted before the GPU smokes have
actually passed. The experiment is complete only when the parent suite's
`intervention_comparison/index.json` exists, contains `"complete": true`, and
was produced after all prompt, row-manifest, probe, layer, zero-output, and
telemetry checks. Pending jobs, launchers, or partial method directories are
not experimental results.

Monitor the recorded IDs with the new cluster's normal `squeue`/`sacct`
commands. The machine-readable terminal check is:

```bash
test -f "$RUN_ROOT/artifacts/concept_intervention/\
qwen35_4b_three_method_intervention_v1/intervention_comparison/index.json"
"$JLENS_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["RUN_ROOT"]) / (
    "artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/"
    "intervention_comparison/index.json"
)
payload = json.loads(path.read_text())
assert payload["complete"] is True, payload
print(path)
PY
```

Batch jobs run artifact paths relative to the persistent `RUN_ROOT`. Committed
read-only resources such as the frozen prompt banks resolve through the
explicit `JLENS_REPOSITORY_ROOT=CODE_ROOT` fallback; this fallback is confined
to that checkout and does not redirect missing artifact paths.
Recovery after a solver-code correction must rerun every occupancy shard with
`OCCUPANCY_OVERWRITE=1`; the default remains resumable and never overwrites a
completed shard.
