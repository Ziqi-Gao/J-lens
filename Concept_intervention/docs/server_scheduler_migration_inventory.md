# ServerScheduler protocol-v2 migration inventory and bounded-pilot proposal

This document records the completed J-lens project-side protocol-v2 adapter,
execution-profile, allocation-validation, retry/completion, request-example,
fixture-test, and migration-inventory stages. The legacy scheduler remains
available. No request was submitted, no central
registration was edited or enabled, no service was changed, no experiment or
pilot was launched, and no legacy file or runtime state was deleted.

## Stable foreground entrypoint

Proposed absolute entrypoint:

```text
/home/del6500/projects/J-lens/Concept_intervention/scripts/server_scheduler_entrypoint.sh
```

It accepts only `--job-manifest RUNNING_JOB_JSON`, remains in the foreground,
validates the protocol-v2 running manifest, selected `execution_profile`, final
allocation, durable one-based attempt, and scheduler environment, then
dispatches a fixed allowlist to existing scientific task bodies. It never
executes a command from `parameters`, selects a GPU, polls `nvidia-smi`, creates
a project GPU lock or lease, publishes a job, or creates Screen/tmux sessions.
The foreground runner enters through the new scheduler-only
`server_scheduler_three_method_task.sh` and `server_scheduler_kdiag_task.sh`
transport wrappers. This migration leaves registered historical
launcher/protocol paths byte-for-byte unchanged relative to the checked-out
HEAD; the wrappers delegate only read-only legacy task bodies and own the new
write-once derivation routing themselves.

The implementation is split under `src/jlens_workspace/scheduler/`: `catalog`
owns the 28-task migration inventory and profiles; `manifest`, `allocation`, and `environment` own
the fail-closed v2 input boundary; `runner` owns fixed foreground dispatch;
`completion` owns scientific gates and receipts; and `dag` owns the
side-effect-free dependency specification. The historical
`server_scheduler_adapter.py` path is a compatibility facade. Tests require the
one-task TOML proposal to be an exact safe subset of the internal catalog and
require its request fixture to match that allowlist.

The adapter consumes the plural v2 GPU identities
`SERVER_SCHEDULER_GPU_INDICES`, `SERVER_SCHEDULER_GPU_UUIDS`, and
`SERVER_SCHEDULER_GPU_PCI_BUS_IDS`; it rejects all singular v1 identity
variables. CPU jobs require empty plural GPU values and no CUDA visibility.
GPU jobs preserve the scheduler's `CUDA_VISIBLE_DEVICES` order, so the task sees
logical CUDA device zero and never reinterprets it as a host index. The
utilization reservation and concrete `exclusive`/`shared` result are also
checked against the manifest.

Allocation validation follows the central NUMA contract: a placement confined
to one NUMA node requires matching integer `numa_node` and
`SERVER_SCHEDULER_NUMA_NODE` values, while a cross-NUMA placement requires
`numa_node: null` and omission of `SERVER_SCHEDULER_NUMA_NODE`. A stale NUMA
environment value fails closed.

Before dispatch, the adapter derives the code root from the current checkout,
runs a non-locking porcelain status check, and
rejects any tracked or untracked checkout change. The manifest's
`parameters.git_commit` therefore identifies the code that can actually run;
the current dirty checkout cannot be piloted until the migration work and its
dependencies are committed. No new worktree is used by this migration.

The task remains in the foreground. A nonzero scientific exit code is returned
unchanged and creates no completion receipt. After a zero exit, the adapter
validates the task-specific scientific marker or output manifest, writes
`/scr/del6500/J-lens/server_scheduler/completions/<job_id>.json` atomically,
re-reads it, and only then returns zero. The receipt binds the job, task, run,
commit, shard, stage/instance, selected profile, completed attempt, and hashes
of immutable validated metadata. On at-least-once redelivery, the adapter
revalidates the scientific evidence and receipt before returning zero without
rerunning the task. Read-only preflight tasks have no durable scientific output,
so their atomic receipt is written only after their command exits zero. The
K-diagnostic resource gate is refreshed after every bounded wave; it must still
exist with the matching stage and `approved: true` on every receipt check, but
its deliberately mutable file hash is not embedded in an earlier wave's
permanent receipt.

The internal catalog deliberately classifies 23 tasks as `sealed_source`, three as
`derivation`, and two as `read_only`. `sealed_source` tasks are rejected before
process launch because their registered design roots are already complete and
immutable. The derived tasks publish only to typed `revision-rN` roots by
write-once sibling staging and atomic directory rename; the read-only tasks may
inspect but not mutate scientific artifacts. The machine proposal does **not**
register that 28-task inventory. It exposes only `kdiag-validate`, a CPU-only,
single-shard, read-only task. The 23 sealed nodes, all three derivations, and
the other read-only task are absent, so attempt/adoption routing is not an
activation blocker for this bounded pilot.

The checked-in `enabled = false` is only the protocol-required project-to-central
handoff default. The integration itself is pilot-ready: after a real clean
commit and exact request are verified, a ServerScheduler-owned operation may
install this one-task registration disabled and separately enable exactly this
bounded pilot. It must not substitute or enable the internal 28-task catalog.

The proposed registration metadata is:

```text
schema_version = 2
name = "J-lens"
enabled = false
integration_status = "protocol-v2-bounded-read-only-pilot-ready"
code_root = "/home/del6500/projects/J-lens"
data_root = "/data/del6500/J-lens"
scratch_root = "/scr/del6500/J-lens"
proposed_handoff_root = "/scr/del6500/J-lens/scheduler"
allow_cpu_jobs = true
allow_gpu_jobs = false
allow_process_signals = false
allow_project_file_mutation = false
```

The complete machine-readable minimal proposal is
`Concept_intervention/docs/server_scheduler_registration_proposal.toml`. A
fixture test requires its single task, profile, runtime estimate, and entrypoint
to match the internal adapter catalog. ServerScheduler's own v2 registration
loader parses it as a disabled one-task handoff registration with a CPU-only
policy. A
ServerScheduler-owned session must still review and install it; this J-lens
session does not write central configuration.

## Registration proposal

The actual pilot registration contains exactly one task:

| Task | Profile | CPU min/pref/max | Goal | Host MiB | GPU | Output policy | Initial seconds |
| --- | --- | ---: | --- | ---: | ---: | --- | ---: |
| `kdiag-validate` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | read-only | 60 |

It validates the pilot K-diagnostic configuration and exits in the foreground.
It neither loads a model nor writes scientific data. GPU jobs are disabled for
this registration.

### Internal non-pilot catalog inventory

All envelopes are fixed because the available legacy observations do not
establish safe CPU scaling. CPU values are physical cores, memory is MiB, GPU
memory/utilization is per device, and `cpu_scaling_efficiency = 0.0`. CPU
profiles use the v2-required `gpu_exclusivity = "shareable"`; this describes
the absence of a GPU, not CPU co-tenancy policy. Every GPU profile uses
`gpu_exclusivity = "required"`: old local occupancy sharing is migration
evidence, not a central interference measurement.

| Task | Profile | CPU min/pref/max | Goal | Host MiB | GPU × per-device MiB/util | Exclusivity | Initial seconds |
| --- | --- | ---: | --- | ---: | --- | --- | ---: |
| `preflight` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 60 |
| `lens` | `gpu-24g-required` | 4/4/4 | balanced | 24,576 | 1 × 24,576/40% | required | 21,600 |
| `capture` | `gpu-24g-required` | 4/4/4 | balanced | 24,576 | 1 × 24,576/40% | required | 900 |
| `layer-selection` | `cpu-16-fixed` | 16/16/16 | balanced | 32,768 | 0 | shareable | 900 |
| `bootstrap-probes` | `cpu-8-fixed` | 8/8/8 | throughput | 16,384 | 0 | shareable | 120 |
| `bootstrap-index` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 60 |
| `occupancy` | `gpu-2g-required` | 4/4/4 | throughput | 4,096 | 1 × 2,048/20% | required | 4,200 |
| `occupancy-index` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 60 |
| `iti-capture` | `gpu-24g-required` | 4/4/4 | balanced | 24,576 | 1 × 24,576/40% | required | 900 |
| `iti-fit` | `cpu-16-fixed` | 16/16/16 | balanced | 32,768 | 0 | shareable | 120 |
| `j-grid` | `gpu-24g-required` | 4/4/4 | throughput | 24,576 | 1 × 24,576/40% | required | 5,400 |
| `raptor-grid` | `gpu-24g-required` | 4/4/4 | throughput | 24,576 | 1 × 24,576/40% | required | 5,400 |
| `iti-grid` | `gpu-24g-required` | 4/4/4 | throughput | 24,576 | 1 × 24,576/40% | required | 5,400 |
| `smoke-check` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 60 |
| `method-index` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 1,800 |
| `comparison-index` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 120 |
| `candidate-rescore` | `gpu-24g-required` | 4/4/4 | throughput | 24,576 | 1 × 24,576/40% | required | 14,400 |
| `comparison-rescore-index` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 180 |
| `kdiag-validate` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 60 |
| `kdiag-prepare-targets` | `gpu-24g-required` | 4/4/4 | balanced | 24,576 | 1 × 24,576/40% | required | 120 |
| `kdiag-build-bases` | `gpu-24g-required` | 4/4/4 | balanced | 24,576 | 1 × 24,576/40% | required | 120 |
| `kdiag-rotation-cache-preflight` | `cpu-16-fixed` | 16/16/16 | balanced | 32,768 | 0 | shareable | 120 |
| `kdiag-prepare-rotations` | `cpu-16-fixed` | 16/16/16 | balanced | 32,768 | 0 | shareable | 900 |
| `kdiag-resource-preflight` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 60 |
| `kdiag-index` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 1,200 |
| `kdiag-report` | `cpu-4-fixed` | 4/4/4 | balanced | 8,192 | 0 | shareable | 300 |
| `kdiag-microbenchmark` | `gpu-4g-required` | 4/4/4 | throughput | 12,288 | 1 × 4,096/20% | required | 21,600 |
| `kdiag-bundle` | `gpu-4g-required` | 4/4/4 | throughput | 12,288 | 1 × 4,096/20% | required | 7,200 |

The GPU rows below are retained only as unregistered migration evidence. Every
such internal task allowlists only
`NVIDIA RTX PRO 6000 Blackwell Server Edition`. The initial runtimes are rounded
conservative legacy-local baselines informed by completed task logs. Legacy log
file lifetimes include local admission waiting and therefore are not treated as
clean runtime measurements. They are not part of the bounded pilot proposal and
cannot be requested through it. No alternate CPU/GPU profiles are proposed
because scientific equivalence and scaling have not been measured.

## Parameter allowlists

Three-method tasks accept exactly:

```json
{
  "run_id": "qwen35_4b_three_method_intervention_v1",
  "git_commit": "40-lowercase-hex",
  "shard_index": 0,
  "shard_count": 1
}
```

The fixed shard counts are: `bootstrap-probes=7`, `occupancy=35`,
`j-grid=392`, `raptor-grid=63`, `iti-grid=3087`, `method-index=3`, and
`candidate-rescore=3`; all other three-method tasks use one shard.

K-diagnostic tasks accept exactly the four fields above plus:

```json
{
  "stage": "pilot|full|transformed",
  "instance": "strict identifier"
}
```

Task/stage combinations are fixed in the adapter. Bundle and microbenchmark
requests must match the persisted `bundles.json` count; the microbenchmark
index must be the maximum registered conservative-cost bundle. All
project-produced requests use priority zero. Priority is an operator policy;
J-lens does not self-declare an emergency. No parameter may contain a command,
executable, path override, environment override, credential, token, or secret.

The formal bounded-pilot request path is:

```text
/scr/del6500/J-lens/scheduler/outbox/jlens-kdiag-validate-bounded-pilot.json
```

Writing a request there is not submission or completion evidence. Only a
ServerScheduler-owned session may validate and submit that exact file. It must
be created only after the main checkout has a clean immutable commit, using
that real 40-hex commit. The checked-in pilot template is:

```text
/home/del6500/projects/J-lens/Concept_intervention/docs/server_scheduler_request_kdiag_v2.example.json
```

Its all-zero 40-hex `git_commit` is an explicit template sentinel and is never
publishable. The formal outbox uses `task = "kdiag-validate"`, `stage =
"pilot"`, one shard, priority zero, and no `resources` or forced
`execution_profile`. The retained three-method example is a non-pilot catalog
fixture and is not accepted by this minimal registration.

Before any write, the adapter validates that the run, derivation, and receipt
paths remain inside the J-lens-owned data/scratch trees and contain no symlink
component. There is no parameter-controlled output path. CPU worker and
BLAS/OpenMP counts are set from the allocation. Derived task bodies write to
sibling attempt-staging directories, write `index.json` last, and atomically
publish only a complete final root. A missing scratch receipt can safely adopt
an already valid derived index. A partial/conflicting final root fails closed
and requires a registered same-revision attempt/recovery operation; staging
debris is preserved for audit and never mislabeled as a new revision. Central
redelivery does not use legacy `.done` files as scientific evidence.

## Scientific DAG submission requirements

The side-effect-free project DAG specification now lives in
`src/jlens_workspace/scheduler/dag.py`. It expands every fixed three-method
shard exactly once, preserves the smoke and rescore gates, and expands each
K-diagnostic stage from the content-addressed bundle count/worst-bundle pair
with a mandatory resource preflight after every bounded wave. It validates
catalog membership, shard counts, stage/instance identities, duplicate
invocations, missing dependencies, and cycles. It neither writes an outbox nor
submits work.

This is the preserved scientific DAG, not the bounded-pilot registration. The
minimal registration exposes only `kdiag-validate`; none of its 23
`sealed_source` nodes or three derivations can be submitted through that
registration. A future full-DAG publisher must first map sealed nodes to registered
same-design attempt roots or read-only adoption jobs without weakening any
dependency or completion gate. That future routing is not required to run the
single read-only pilot.

A future approved outbox publisher must consume that validated specification
and preserve these dependencies rather than submitting all allowlisted tasks
at once.

Three-method DAG:

```text
preflight
-> lens + capture
-> layer-selection
-> bootstrap-probes[0..6] -> bootstrap-index
-> iti-capture -> iti-fit
-> occupancy[0..34] -> occupancy-index
-> j-grid[6] + raptor-grid[8] + iti-grid[6,251]
-> smoke-check and validated smoke gate
-> remaining j-grid[0..391], raptor-grid[0..62], iti-grid[0..3086]
-> method-index[0..2]
-> comparison-index
-> candidate-rescore[0..2]
-> comparison-rescore-index
```

The base comparison gate remains `intervention_comparison/index.json` with
`"complete": true`. The derived revision-r2 method indexes live under
`derivations/revision-r2/candidate_score_rescore_v1/<method>/index.json`, and
the final rescore gate is
`derivations/revision-r2/intervention_comparison/index.json`. Each comparison
index also pins and validates its companion `comparison.json`. Scheduled
rescoring never passes `--overwrite` and never rewrites the base comparison.
An outbox file, central queue state, process exit, or legacy `.done` marker is
not a scientific completion marker. The separate `finalize` and `rescore`
dependency rules must remain intact if their publishers are later implemented.

K-diagnostic DAG:

```text
validate stage
-> prepare-targets (pilot/full) or build-bases (transformed)
-> rotation-cache-preflight -> prepare-rotations
-> worst-bundle microbenchmark -> resource-preflight
-> bounded bundle waves with resource-preflight after every wave
-> stage index
pilot index -> full index -> transformed index -> report
```

Every stage index must retain its existing `"complete": true` audit. The
resource-preflight and worst-bundle gates remain mandatory.

Task-level completion validation does not weaken those DAG gates:

- lens, activation capture, layer selection, bootstrap, occupancy, and ITI-fit
  tasks validate their fixed manifests/metadata and required companion files;
- each intervention grid shard validates its exact concept/grid summary and
  required generation, blind-map, and candidate-score files;
- smoke, method, rescore, comparison, occupancy, and bootstrap index tasks
  require their existing atomic `complete: true` indexes;
- K-diagnostic target/basis/rotation indexes require their existing completion
  contracts, both preflights require `approved: true`, bundles require the
  exact `bundles.json` identity and bundle `complete.json` (including the
  microbenchmarked worst bundle), and stage indexes remain `complete: true`.
  Reporting publishes `derivations/revision-r1/report/index.json` last and
  validates every indexed report, decision, plot-data, figure, and immutable
  source descriptor hash. The per-wave resource-preflight file is an operational
  mutable gate that is revalidated but intentionally not receipt-hash evidence;
- legacy occupancy metrics contain producer-written non-finite JSON values.
  Only this known scientific evidence reader permits that historical encoding;
  central manifests, requests, and project completion receipts remain strict
  JSON.

The future DAG publisher must query central job status and then validate these
scientific gates before releasing dependants. The adapter's per-job receipt
solves at-least-once execution; it is not a replacement for a stage or final
DAG completion index.

## Legacy inventory and later deletion candidates

Executable callers and orchestration references found in the checkout:

- `three_method_local_worker.sh` calls both direct resource wrappers.
- `run_three_method_local.sh` owns the three-method dependency graph, local
  `dag.lock`, xargs worker pools, status command, smoke-marker prerequisite,
  and final `complete=true` check.
- `run_qwen35_4b_k_diagnostic_v2_local_worker.sh` calls both wrappers.
- `run_qwen35_4b_k_diagnostic_v2.sh` owns K-diagnostic stage dependencies,
  `dag.lock`, waves, resource-preflight cadence, status command, and index
  prerequisites.
- `launch_three_method_local_screen.sh` creates detached Screen controllers.
- `jlens-three-method.service` calls
  `/home/del6500/.local/libexec/jlens-three-method-service`, which in turn
  calls `run_three_method_local.sh full`.
- `jlens-three-method-judge-v1.service` directly calls a snapshot copy of
  `three_method_local_resources.sh` around the historical judge controller.

Allocation and provenance references found:

- `k_diagnostic/experiment.py` accepted the local lease file or Slurm; the
  adapter stage adds an independently validated ServerScheduler running
  manifest while retaining both legacy backends.
- `tests/test_local_scheduler.py` covers the old broker, GPU observation,
  locks, leases, and resource selection.
- `tests/test_local_three_method_scripts.py` covers old profiles, worker
  routing, task decoding, and scientific DAG text.
- `tests/test_k_diagnostic.py` and `tests/test_cli.py` contain historical
  `fsm_local_scheduler` provenance fixtures that remain valid compatibility
  inputs.
- `AGENTS.md`, `Concept_intervention/docs/local_server.md`, and
  `Concept_intervention/docs/qwen35_4b_k_diagnostic_v2.md` describe the legacy
  scheduler. They must be revised at dependency cutover, not during this
  adapter-only stage.

Runtime inspection was intentionally read-only. The sandbox could not access
the host PID namespace, `/run/screen`, or the user-systemd bus, so no claim of
process/session/service inactivity is made. Both unit files and both
`default.target.wants` symlinks exist. The three-method service wrapper still
pins commit `356482be4192fd0ee5eb7e2fc74b1f12f1ecf501`, while the inspected
checkout is at a different commit; this explains a preflight failure but is
not proof that the service is inactive.

Any future retirement action first requires a central pilot, drain, dependency
cutover, clean rollback point, live-reference audit, and explicit deletion
approval. No source file currently meets those gates, and nothing was deleted
in this stage.

Exact source deletion candidates at this stage: **none**.

Under the current immutable-compatibility policy, the following files are a
dependency-closure **retention set**, not deletion candidates. Registered
historical launchers still reach them directly or transitively:

- `src/jlens_workspace/local_scheduler.py`
- `Concept_intervention/scripts/three_method_local_gpu.sh`
- `Concept_intervention/scripts/three_method_local_resources.sh`
- `tests/test_local_scheduler.py`
- `Concept_intervention/scripts/three_method_local_worker.sh`

An ordinary central cutover or caller audit is insufficient to remove any item
in that retention set. Reconsideration would first require a separate,
explicitly approved launcher-compatibility policy migration that defines how
the immutable historical runtime is preserved, introduces a versioned
replacement without rewriting a registered path, and proves that no registered
launcher, process, service, or rollback route depends on the candidate. Until
then all five paths remain in place.

The following exact paths are also *not* deletion candidates because completed
experiment manifests register them as immutable compatibility/provenance
inputs:

- `Concept_intervention/scripts/launch_three_method_local_screen.sh`
- `Concept_intervention/scripts/run_three_method_local.sh`
- `Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v2.sh`
- `Concept_intervention/scripts/run_qwen35_4b_k_diagnostic_v2_local_worker.sh`

Central cutover must not replace their behavior in place. New transport belongs
in a new versioned or scheduler-specific path, as implemented by the two
`server_scheduler_*_task.sh` wrappers in this migration.

The unchanged `run_three_method_local_task.sh` and
`run_qwen35_4b_k_diagnostic_v2_local_task.sh` task bodies remain historical
dependencies, not deletion candidates; the new scheduler wrappers delegate
only their read-only branches. `three_method_local_paths.sh` and
`three_method_local_env.sh` retain project path/environment responsibilities
and are not scheduler-only files.

Exact scratch retirement candidates observed on 2026-08-31:

- `/scr/del6500/J-lens/runtime/three-method/scheduler`
- `/scr/del6500/J-lens/runtime/three-method/gpu-locks`
- `/scr/del6500/J-lens/runtime/three-method/dag.lock`
- `/scr/del6500/J-lens/runtime/three-method/k-diagnostic-v2/dag.lock`
- `/scr/del6500/J-lens/runtime/three-method/state`
- `/scr/del6500/J-lens/runtime/three-method/k-diagnostic-v2/state`
- `/scr/del6500/J-lens/logs/three-method/screen`

The lease directory contained zero JSON leases at inspection time; 20 legacy
GPU lock files existed. Lock-file presence alone does not prove inactivity.
Move exact inactive roots to a dated J-lens retirement directory before any
later deletion. Durable scientific artifacts and completion indexes are never
cleanup targets.

`git worktree list` reports four pre-existing linked worktrees. Two are
correctly below `/scr/del6500/J-lens/worktrees/`; two legacy checkouts are under
the now-forbidden
`/home/del6500/projects/J-lens/.git/codex-worktrees/{llm-judge,three-method-llm-judge-v1}`.
All four were clean in a read-only status audit, but no worktree was created or
removed here. Cleanliness alone is insufficient: branch refs, live
process/service paths, and durable artifacts must also be preserved and
audited. Any later removal must use `git worktree remove`, never recursive
deletion.

External operator targets, outside the J-lens write boundary:

- `/home/del6500/.config/systemd/user/jlens-three-method.service`
- `/home/del6500/.config/systemd/user/default.target.wants/jlens-three-method.service`
- `/home/del6500/.local/libexec/jlens-three-method-service`
- `/home/del6500/.config/systemd/user/jlens-three-method-judge-v1.service`
- `/home/del6500/.config/systemd/user/default.target.wants/jlens-three-method-judge-v1.service`

Only a separately approved operator session may stop/disable units, verify
inactivity, remove exact files, reload user systemd, and re-audit references.

## Verification

Offline checks run from the J-lens checkout on 2026-09-01:

- Ordinary suite: `376 passed, 1 skipped`, with four upstream scikit-learn
  deprecation warnings; the skip is the unavailable pinned RAPTOR checkout.
- Focused ServerScheduler adapter suite: `34 passed`, covering strict v2
  manifests, requested-versus-concrete allocation, plural GPU identity,
  cross-NUMA placement, legacy environment sanitization, owned non-symlink
  paths, sealed-source refusal, staged derivation adoption/recovery, canonical
  K-diagnostic stage identities, scheduler-specific transport paths, and full
  companion-hash completion chains. The adapter-plus-legacy-script boundary
  suite passed `46` tests and verifies that registered historical task paths do
  not admit the scheduler environment and that the read-only pilot bypasses
  legacy resource orchestration.
- Full Ruff check: passed.
- All checked-in shell/Slurm `bash -n`: passed.
- `git diff --check`: passed.
- Experiment registry validation: `10 designs` valid both normally and with
  `--check-artifacts`; typed revisions and J-lens-owned artifact/runtime roots
  are validated.
- ServerScheduler's own registration loader parsed the machine proposal as
  `J-lens`, a disabled handoff with exactly one CPU-only task and the expected
  entrypoint.
- ServerScheduler's v2 request model parsed the checked-in bounded-pilot
  template with no resource or forced-profile override.
- Direct execution of the proposed `kdiag-validate` body from its real durable
  run root completed read-only validation of all `14` pinned inputs. This was
  an offline validation check, not a submitted central pilot.
- A read-only audit validated existing durable evidence for 25 of the 28 task
  contracts. The three expected missing contracts are the unexecuted
  revision-r2 candidate rescore, revision-r2 rescore comparison, and
  revision-r1 K-diagnostic report. The audit did not recompute or modify an
  artifact.

No GPU, remote, scientific, or bounded central pilot was run.

## Blockers and rollback

- The 23 `sealed_source` tasks remain a **future full-DAG** gap: every sealed
  node needs a registered same-design `attempt-NNN` output root or an explicit
  read-only adoption task before full-DAG activation. They are absent from the
  one-task pilot proposal and do not block that pilot.
- The three typed derivations are `registered_not_executed`: three-method
  revision-r2 candidate rescoring, its revision-r2 comparison, and
  K-diagnostic revision-r1 reporting have no final durable index yet. This is
  why the read-only artifact audit is 25/28 rather than 28/28.
- The v2 adapter, entrypoint, minimal proposal, examples, inventory, and tests
  now exist in clean immutable commit
  `6713a3eeeed117ffb94fa5903f293f32564d3752` on
  `codex/three-method-interventions`. This satisfies the bounded pilot's code
  identity gate; it does not authorize full-DAG activation.
- The installed central registration at
  `/home/del6500/projects/ServerScheduler/config/projects/j-lens.toml` remains
  disabled with `tasks = []` and an empty entrypoint. The checked-in J-lens
  proposal is parseable and contains only the reviewed `kdiag-validate` pilot,
  but it has not been installed; this project session cannot edit
  ServerScheduler registration.
- The formal bounded-pilot outbox request now exists at
  `/scr/del6500/J-lens/scheduler/outbox/jlens-kdiag-validate-bounded-pilot.json`
  and is regenerated against the final clean immutable project HEAD. It
  contains only
  `kdiag-validate`, stage `pilot`, shard `0/1`, priority zero, and no resource
  or forced-profile override. Writing this file is not central submission.
- The single pilot does not require a DAG publisher: a ServerScheduler operator
  can validate and submit the exact outbox file. A future full DAG still needs
  an approved publisher/consumer; the project must not call
  `server_scheduler submit` itself.
- ServerScheduler's deployment record says its canonical user service was
  installed, enabled, active, and running when last verified on 2026-08-31.
  This J-lens sandbox cannot attest its live state. The central J-lens
  registration is still inventory-only and disabled, so a separate
  ServerScheduler-owned session must install the reviewed one-task proposal
  disabled and then enable exactly that task for the approved pilot.
- GPU and shared-occupancy profiles remain unregistered. The bounded pilot uses
  only the conservative fixed CPU envelope and calibrates its runtime without
  admitting any GPU work.
- This sandbox cannot inspect the host PID, Screen, or user-systemd runtime
  namespaces. Unit files and enablement symlinks exist, but live/inactive state
  requires a separate operator audit.
- No bounded central pilot has run, so exit status, artifact hashes, resume,
  resource use, and failure behavior have not been compared with the legacy
  path.

The observed legacy rollback candidate is commit
`90b0b984f2305002eaf3ce8e63eb19595731d035` on
`codex/three-method-interventions`; it contains the still-present legacy
scheduler. No rollback branch/ref was created because `.git` is protected and
the checkout contains unrelated dirty work. Before cutover, preserve that
commit with a clean branch/ref and record an immutable runtime snapshot.

Rollback must first disable central intake and drain central jobs. The current
legacy unit and service wrapper hard-code the main checkout at
`/home/del6500/projects/J-lens`; creating a rollback worktree below
`/scr/del6500/J-lens/worktrees/` alone therefore does **not** activate it. A
second guard in the wrapper requires branch
`codex/three-method-interventions` at commit
`356482be4192fd0ee5eb7e2fc74b1f12f1ecf501`; it would reject the observed
`90b0b984f2305002eaf3ce8e63eb19595731d035` rollback candidate even at the
correct path. A separately approved Git/operator maintenance operation must
choose exactly one audited activation path:

1. after preserving or resolving every current dirty-tree change, place the
   main checkout itself at the immutable rollback ref so the unit's hard-coded
   paths resolve to that code, then update the wrapper's expected branch and
   commit to that exact audited ref; or
2. create a clean worktree at the immutable rollback ref below the approved
   scratch worktree root, then change the exact user-systemd unit and service
   wrapper paths **and** the wrapper's expected branch/commit guards to that
   worktree and ref.

For either service path, the operator must run the wrapper's `--check`, reload
user systemd after unit changes, and re-audit every executable,
working-directory, branch, and commit reference before start. The alternative
is to leave the service disabled and invoke the audited legacy launcher
manually from the clean rollback checkout; that choice must not reuse or imply
successful service preflight.

Neither path is executable from the current protected, dirty J-lens session.
Resume requires explicit operator approval and a fresh live-process/service
audit. Rollback must never overwrite, reinterpret, or delete durable scientific
artifacts.
