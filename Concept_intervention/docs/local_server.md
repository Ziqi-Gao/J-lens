# Local non-Slurm server runbook

This runbook adapts the registered three-method intervention experiment to the
managed four-GPU server. It changes execution and storage only. The four formal
YAML files, scientific identities, grid indices, method implementations, and
artifact validation contracts remain unchanged.

## Project boundary and storage

J-lens and FedFisher are independent projects. A J-lens agent may read the
FedFisher maintenance guide for server policy, but a single agent/task/session
must never write both projects.

| Root | Responsibility |
| --- | --- |
| `/home/del6500/projects/J-lens` | Git checkout, tests, configs, scripts, docs |
| `/data/del6500/J-lens` | prepared data, fitted lens, generations, durable indexes |
| `/scr/del6500/J-lens` | Python environment, download/build caches, locks, logs, runtime state |

The registered local defaults are:

```text
CODE_ROOT=/home/del6500/projects/J-lens
RUN_ROOT=/data/del6500/J-lens/runs/qwen35_4b_three_method_intervention_v1
JLENS_PYTHON=/scr/del6500/J-lens/envs/three-method/bin/python
HF_HOME=/scr/del6500/J-lens/cache/huggingface
JLENS_LOCAL_RUNTIME_ROOT=/scr/del6500/J-lens/runtime/three-method
JLENS_LOCAL_LOG_ROOT=/scr/del6500/J-lens/logs/three-method
```

`three_method_local_paths.sh` rejects code, data, environment, cache, runtime,
or temporary overrides that escape the three J-lens roots. Do not point any
J-lens variable at a FedFisher directory.

## 1. Create the environment

The server provides `/usr/bin/python3.12` only to bootstrap uv. The setup
entrypoint then installs a complete uv-managed CPython 3.12, including build
headers needed by Triton, and creates the locked project environment under
J-lens scratch. It does not install a system or user-level package and does
not download model weights or datasets.

```bash
cd /home/del6500/projects/J-lens
Concept_intervention/scripts/setup_three_method_local.sh
```

The command performs `uv sync --frozen --extra dev --extra llm` against that
managed interpreter. Its Python distribution, tool environment, wheel, Git
dependency, Torch, Triton, and compiler caches remain under
`/scr/del6500/J-lens`. The recorded environment summary is:

```text
/scr/del6500/J-lens/runtime/three-method/environment.txt
```

## 2. Review and freeze the local adaptation

Scientific execution requires a clean immutable checkout. Review the local
adaptation, run offline tests, and create an intentional successor commit before
bootstrap. The runtime records that exact commit; a later code change gets a
separate state namespace and cannot silently inherit completion markers.

The local execution plan is inspectable without an environment or GPU:

```bash
Concept_intervention/scripts/run_three_method_local.sh plan
```

Never launch `run_three_method_local_task.sh` or
`three_method_local_worker.sh` directly. They intentionally reject execution
outside the top-level DAG.

## 3. Bootstrap pinned inputs once

Bootstrap is the only online scientific-input phase:

```bash
Concept_intervention/scripts/launch_three_method_local_screen.sh bootstrap
```

It downloads and verifies:

- `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- GoEmotions at `add492243ff905527e67aeb8b80c082af02207c3`;
- external RAPTOR at `cf7405899174af39f3970e093e4b86bf0972ff87`; and
- the installed official J-lens source at
  `581d398613e5602a5af361e1c34d3a92ea82ba8e`.

Prepared source data and every formal artifact start under `RUN_ROOT`; no
artifact from another server or project is accepted. After bootstrap, task
environments force Hugging Face, Transformers, and Datasets offline.

## 4. GPU overlay policy

The user has authorized J-lens to run alongside pre-existing GPU work when
there is stable spare capacity. The local wrapper samples each candidate GPU
three times and takes a J-lens-only `flock` before launch. It records free
memory, utilization, and the existing compute-process snapshot, then exposes
exactly one physical GPU with `CUDA_VISIBLE_DEVICES`.

Defaults are intentionally conservative:

```bash
export JLENS_LOCAL_GPU_IDS=0,1,2,3
export JLENS_LOCAL_GPU_MIN_FREE_MIB=65536
export JLENS_LOCAL_GPU_MAX_UTILIZATION=70
export JLENS_LOCAL_GPU_ALLOW_OVERLAY=1
export JLENS_LOCAL_GPU_STABILITY_SAMPLES=3
export JLENS_LOCAL_GPU_SAMPLE_INTERVAL_SECONDS=2
export JLENS_LOCAL_GPU_POLL_SECONDS=30
export JLENS_LOCAL_GPU_WAIT_TIMEOUT_SECONDS=0
export JLENS_LOCAL_GPU_WORKERS=4
export JLENS_LOCAL_CPU_WORKERS=4
```

A timeout of zero means wait without disturbing existing work. A foreign PID is
never killed, paused, signalled, inspected with a debugger, or reconfigured.
Setting `JLENS_LOCAL_GPU_ALLOW_OVERLAY=0` makes the gate require no existing
compute process. Lowering the free-memory threshold is an operator decision and
must be based on measured smoke memory, not merely on model-weight size.

J-lens locks coordinate only J-lens workers; they do not claim ownership over
another project's GPU process. If an overlay fails or interferes, stop or
adjust only the exact J-lens Screen session/PID after confirming its command
and ownership.

## 5. Run the smoke DAG

```bash
Concept_intervention/scripts/launch_three_method_local_screen.sh smoke
```

The top-level order is:

```text
preflight
-> J-lens refit + shared resid_post capture
-> RAPTOR probes / shared Top-6
-> bootstrap probes
-> ITI head capture / fit
-> J occupancy
-> J[6], RAPTOR[8], ITI[6], ITI[251]
-> smoke-check
```

J-lens refit and residual capture may occupy two capacity-approved GPUs.
Array-like stages use bounded local worker pools. A per-task marker is written
only after the command exits successfully:

```text
/scr/del6500/J-lens/runtime/three-method/state/<git-commit>/*.done
```

Markers support restart and are not scientific evidence. The smoke gate still
revalidates generation counts, telemetry, identities, hashes, and zero
intervention consistency.

## 6. Run the full grids

Only after the smoke marker and artifact gate exist:

```bash
Concept_intervention/scripts/launch_three_method_local_screen.sh full
```

The local runner omits the four completed smoke shards and preserves the exact
formal array mapping:

- J-component: 392 total shards, smoke index 6;
- RAPTOR: 63 total shards, smoke index 8;
- ITI: 3,087 total shards, smoke indices 6 and 251.

Methods run as separate grid waves. They do not import or invoke each other's
core logic. After all shards finish, the runner builds the three method indexes
and then the comparison index.

For a single uninterrupted controller, `all` runs bootstrap, smoke, and full
in sequence. Prefer separate phases for the first server run so model/CUDA
compatibility and telemetry storage growth can be reviewed at the smoke gate.

## 7. Monitor and complete

```bash
screen -ls
Concept_intervention/scripts/run_three_method_local.sh status
```

Task logs are under:

```text
/scr/del6500/J-lens/logs/three-method/<git-commit>/
```

A Screen session or local `.done` marker is never the final completion signal.
The only final signal remains:

```text
/data/del6500/J-lens/runs/qwen35_4b_three_method_intervention_v1/
  artifacts/concept_intervention/qwen35_4b_three_method_intervention_v1/
  intervention_comparison/index.json
```

That file must exist and contain `"complete": true`. Its producer recursively
validates the prompt and row manifests, probes, selected layers, ITI directions,
generation counts, candidate scores, token telemetry, zero-output equivalence,
provenance, and artifact hashes.
